import os
import tkinter as tk
from tkinter import filedialog
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from math import log10, atan2, degrees
from scipy import fftpack
from scipy.stats import chisquare
import exifread
import cv2
import time


class DetectorManipulacaoAvancado:
    def __init__(self):
        self.benford_teorico = [log10(1 + 1 / d) for d in range(1, 10)]

    # 1. ANÁLISE DA LEI DE BENFORD
    def analisar_lei_benford(self, imagem_path, metodo='dct'):
        """Analisa a distribuição dos primeiros dígitos usando a Lei de Benford"""
        try:
            imagem = cv2.imread(imagem_path, cv2.IMREAD_GRAYSCALE)
            if imagem is None:
                return None

            if metodo == 'dct':
                dados = self._extrair_coeficientes_dct(imagem)
            elif metodo == 'pixels':
                dados = self._extrair_valores_pixels(imagem)
            elif metodo == 'gradientes':
                dados = self._extrair_gradientes(imagem)
            else:
                dados = self._extrair_coeficientes_dct(imagem)

            primeiros_digitos = self._extrair_primeiros_digitos(dados)

            if len(primeiros_digitos) == 0:
                return None

            distribuicao_observada = self._calcular_distribuicao(primeiros_digitos)

            correlacao = np.corrcoef(self.benford_teorico, distribuicao_observada)[0, 1]
            distancia_euclidiana = np.sqrt(
                np.sum((np.array(self.benford_teorico) - np.array(distribuicao_observada)) ** 2))

            # Teste estatístico
            try:
                estatistica_chi2, p_valor_chi2 = chisquare(distribuicao_observada,
                                                           [p * len(primeiros_digitos) for p in self.benford_teorico])
            except:
                p_valor_chi2 = 0

            segue_benford = correlacao > 0.95 and p_valor_chi2 > 0.05 and distancia_euclidiana < 0.1

            return {
                'distribuicao_teorica': self.benford_teorico,
                'distribuicao_observada': distribuicao_observada,
                'correlacao': correlacao,
                'distancia_euclidiana': distancia_euclidiana,
                'p_valor_chi2': p_valor_chi2,
                'segue_benford': segue_benford,
                'manipulado': not segue_benford,
                'total_amostras': len(primeiros_digitos)
            }

        except Exception as e:
            print(f"Erro na análise de Benford: {e}")
            return None

    def _extrair_coeficientes_dct(self, imagem):
        """Extrai coeficientes DCT para análise de Benford"""
        coeficientes = []
        try:
            for i in range(0, imagem.shape[0] - 8, 8):
                for j in range(0, imagem.shape[1] - 8, 8):
                    bloco = imagem[i:i + 8, j:j + 8]
                    dct_bloco = cv2.dct(bloco.astype(np.float32))
                    coeficientes.extend(dct_bloco.flatten()[1:20])
        except:
            # Fallback: usar toda a imagem
            dct_total = cv2.dct(imagem.astype(np.float32))
            coeficientes = dct_total.flatten()
        return np.array(coeficientes)

    def _extrair_valores_pixels(self, imagem):
        """Extrai valores de pixels para análise"""
        altura, largura = imagem.shape
        passo = max(1, altura // 100)
        pixels = []

        for i in range(0, altura, passo):
            for j in range(0, largura, passo):
                pixels.append(imagem[i, j])

        return np.array(pixels)

    def _extrair_gradientes(self, imagem):
        """Extrai gradientes da imagem"""
        try:
            grad_x = cv2.Sobel(imagem, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(imagem, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
            return magnitude.flatten()
        except:
            return imagem.flatten()

    def _extrair_primeiros_digitos(self, dados):
        """Extrai o primeiro dígito significativo de cada valor"""
        primeiros_digitos = []

        for valor in dados:
            if valor == 0:
                continue

            valor_abs = abs(valor)

            while valor_abs >= 10:
                valor_abs /= 10
            while valor_abs < 1:
                valor_abs *= 10

            primeiro_digito = int(valor_abs)
            if 1 <= primeiro_digito <= 9:
                primeiros_digitos.append(primeiro_digito)

        return primeiros_digitos

    def _calcular_distribuicao(self, primeiros_digitos):
        """Calcula a distribuição dos primeiros dígitos"""
        if len(primeiros_digitos) == 0:
            return [0] * 9

        contagem = [0] * 9
        for digito in primeiros_digitos:
            if 1 <= digito <= 9:
                contagem[digito - 1] += 1

        total = len(primeiros_digitos)
        return [c / total for c in contagem]

    # 2. DETECÇÃO DE COMPRESSÃO MÚLTIPLA
    def detectar_compressao_multipla(self, imagem_path):
        """Detecta se a imagem foi salva/comprimida múltiplas vezes"""
        try:
            imagem = cv2.imread(imagem_path, cv2.IMREAD_GRAYSCALE)
            if imagem is None:
                return None

            # Análise de histograma
            histograma = cv2.calcHist([imagem], [0], None, [256], [0, 256])
            histograma = histograma.flatten()

            # Calcular entropia do histograma
            probabilidades = histograma / np.sum(histograma)
            entropia = -np.sum([p * np.log2(p) for p in probabilidades if p > 0])

            # Detectar picos no histograma
            picos = self._detectar_picos_histograma(histograma)

            # Análise de blocos
            artefatos_blocos = self._analisar_artefatos_blocos(imagem)

            score_compressao_multipla = (len(picos) / 10 + (8 - min(entropia, 8)) / 2 + artefatos_blocos)

            manipulacao_suspeita = score_compressao_multipla > 1.5

            return {
                'entropia_histograma': entropia,
                'quantos_picos_histograma': len(picos),
                'artefatos_blocos': artefatos_blocos,
                'score_compressao_multipla': score_compressao_multipla,
                'manipulado': manipulacao_suspeita
            }

        except Exception as e:
            print(f"Erro na análise de compressão múltipla: {e}")
            return {
                'entropia_histograma': 0,
                'quantos_picos_histograma': 0,
                'artefatos_blocos': 0,
                'score_compressao_multipla': 0,
                'manipulado': False
            }

    def _detectar_picos_histograma(self, histograma):
        """Detecta picos anormais no histograma"""
        picos = []
        try:
            suavizado = cv2.GaussianBlur(histograma.astype(np.float32), (5, 5), 0)

            for i in range(2, len(suavizado) - 2):
                if (suavizado[i] > suavizado[i - 1] and
                        suavizado[i] > suavizado[i - 2] and
                        suavizado[i] > suavizado[i + 1] and
                        suavizado[i] > suavizado[i + 2] and
                        suavizado[i] > np.mean(suavizado) * 2):
                    picos.append(i)
        except:
            pass
        return picos

    def _analisar_artefatos_blocos(self, imagem):
        """Analisa artefatos de blocos de compressão"""
        try:
            altura, largura = imagem.shape
            artefatos = 0
            total_blocos = 0

            for i in range(8, altura - 8, 8):
                for j in range(8, largura - 8, 8):
                    total_blocos += 1
                    # Verificar descontinuidades nas bordas
                    try:
                        borda_vertical = np.mean(np.abs(imagem[i, j:j + 8] - imagem[i + 1, j:j + 8]))
                        borda_horizontal = np.mean(np.abs(imagem[i:i + 8, j] - imagem[i:i + 8, j + 1]))

                        if borda_vertical > 10 or borda_horizontal > 10:
                            artefatos += 1
                    except:
                        continue

            return artefatos / max(total_blocos, 1)
        except:
            return 0

    # 3. DETECÇÃO DE CLONAGEM OTIMIZADA
    def detectar_clonagem(self, imagem_path):
        """
        Detecta regiões clonadas de forma otimizada
        """
        try:
            print("   Iniciando análise de clonagem...")
            imagem = cv2.imread(imagem_path, cv2.IMREAD_GRAYSCALE)
            if imagem is None:
                return {'clones_detectados': 0, 'manipulado': False, 'metodo': 'rapido'}

            # Redimensionar para análise mais rápida
            if imagem.shape[0] > 600:
                escala = 600 / imagem.shape[0]
                nova_largura = int(imagem.shape[1] * escala)
                imagem = cv2.resize(imagem, (nova_largura, 600))

            altura, largura = imagem.shape
            tamanho_bloco = 32

            # Dividir imagem em blocos
            blocos_por_altura = altura // tamanho_bloco
            blocos_por_largura = largura // tamanho_bloco

            caracteristicas = []

            # Extrair características de cada bloco
            for i in range(blocos_por_altura):
                for j in range(blocos_por_largura):
                    y = i * tamanho_bloco
                    x = j * tamanho_bloco
                    if y + tamanho_bloco <= altura and x + tamanho_bloco <= largura:
                        bloco = imagem[y:y + tamanho_bloco, x:x + tamanho_bloco]

                        # Características simples do bloco
                        media = np.mean(bloco)
                        desvio = np.std(bloco)
                        hist = cv2.calcHist([bloco], [0], None, [4], [0, 256]).flatten()

                        carac = [media, desvio] + hist.tolist()
                        caracteristicas.append(carac)

            if len(caracteristicas) < 2:
                return {'clones_detectados': 0, 'manipulado': False, 'metodo': 'rapido'}

            caracteristicas = np.array(caracteristicas)

            # Encontrar blocos similares
            clones = 0
            for i in range(len(caracteristicas)):
                for j in range(i + 1, min(i + 20, len(caracteristicas))):  # Limitar comparações
                    distancia = np.linalg.norm(caracteristicas[i] - caracteristicas[j])
                    if distancia < 8:  # Limiar de similaridade
                        clones += 1
                        if clones > 10:  # Parar se encontrar muitos clones
                            break

            resultado = {
                'clones_detectados': clones,
                'total_blocos_analisados': len(caracteristicas),
                'manipulado': clones > 5,
                'metodo': 'caracteristicas'
            }

            print(f"   Blocos analisados: {len(caracteristicas)}")
            print(f"   Possíveis clones: {clones}")

            return resultado

        except Exception as e:
            print(f"   ⚠️ ERRO na detecção de clonagem: {e}")
            return {'clones_detectados': 0, 'manipulado': False, 'metodo': 'erro'}

    # 4. ANÁLISE DE ILUMINAÇÃO
    def analisar_iluminacao(self, imagem_path):
        """Analisa a consistência da iluminação"""
        try:
            imagem_color = cv2.imread(imagem_path)
            if imagem_color is None:
                return None

            # Converter para LAB para análise de luminância
            lab = cv2.cvtColor(imagem_color, cv2.COLOR_BGR2LAB)
            luminancia = lab[:, :, 0]

            # Calcular gradientes para estimar direção da luz
            grad_x = cv2.Sobel(luminancia, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(luminancia, cv2.CV_64F, 0, 1, ksize=3)

            # Estimar direção predominante da luz
            magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
            direcao_x = np.mean(grad_x / (magnitude + 1e-8))
            direcao_y = np.mean(grad_y / (magnitude + 1e-8))

            direcao_luz = degrees(atan2(direcao_y, direcao_x))

            # Análise de consistência por quadrantes
            altura, largura = luminancia.shape
            quadrantes = [
                luminancia[:altura // 2, :largura // 2],
                luminancia[:altura // 2, largura // 2:],
                luminancia[altura // 2:, :largura // 2],
                luminancia[altura // 2:, largura // 2:]
            ]

            direcoes_quadrantes = []
            for quadrante in quadrantes:
                try:
                    grad_x_q = cv2.Sobel(quadrante, cv2.CV_64F, 1, 0, ksize=3)
                    grad_y_q = cv2.Sobel(quadrante, cv2.CV_64F, 0, 1, ksize=3)
                    mag_q = np.sqrt(grad_x_q ** 2 + grad_y_q ** 2)
                    dir_x = np.mean(grad_x_q / (mag_q + 1e-8))
                    dir_y = np.mean(grad_y_q / (mag_q + 1e-8))
                    direcoes_quadrantes.append(degrees(atan2(dir_y, dir_x)))
                except:
                    direcoes_quadrantes.append(0)

            inconsistencia_iluminacao = np.std(direcoes_quadrantes)

            return {
                'direcao_luz_principal': direcao_luz,
                'inconsistencia_iluminacao': inconsistencia_iluminacao,
                'manipulado': inconsistencia_iluminacao > 30
            }
        except Exception as e:
            print(f"   ⚠️ ERRO na análise de iluminação: {e}")
            return {
                'direcao_luz_principal': 0,
                'inconsistencia_iluminacao': 0,
                'manipulado': False
            }

    # 5. DETECÇÃO DE RESAMPLING
    def detectar_resampling(self, imagem_path):
        """Detecta redimensionamento usando análise de interpolação"""
        try:
            imagem = cv2.imread(imagem_path, cv2.IMREAD_GRAYSCALE)
            if imagem is None:
                return None

            # Análise no domínio da frequência
            fft = fftpack.fft2(imagem)
            fft_shifted = fftpack.fftshift(fft)
            magnitude = np.log(np.abs(fft_shifted) + 1)

            # Procurar padrões de interpolação
            centro_y, centro_x = magnitude.shape[0] // 2, magnitude.shape[1] // 2

            # Analisar simetria espectral
            try:
                quadrante_superior = magnitude[:centro_y, :centro_x]
                quadrante_inferior = magnitude[centro_y:, :centro_x]

                # Redimensionar para ter o mesmo tamanho
                min_altura = min(quadrante_superior.shape[0], quadrante_inferior.shape[0])
                min_largura = min(quadrante_superior.shape[1], quadrante_inferior.shape[1])

                quadrante_superior = quadrante_superior[:min_altura, :min_largura]
                quadrante_inferior = quadrante_inferior[:min_altura, :min_largura]

                correlacao_vertical = np.corrcoef(quadrante_superior.flatten(),
                                                  quadrante_inferior.flatten())[0, 1]

                if np.isnan(correlacao_vertical):
                    correlacao_vertical = 0
            except:
                correlacao_vertical = 0

            assimetria_espectral = 1 - correlacao_vertical

            return {
                'assimetria_espectral': assimetria_espectral,
                'manipulado': assimetria_espectral > 0.2
            }
        except Exception as e:
            print(f"   ⚠️ ERRO na detecção de resampling: {e}")
            return {
                'assimetria_espectral': 0,
                'manipulado': False
            }

    # 6. ANÁLISE DE METADADOS
    def analisar_metadados(self, imagem_path):
        """Analisa inconsistências nos metadados EXIF"""
        try:
            with open(imagem_path, 'rb') as f:
                tags = exifread.process_file(f, details=False)

            inconsistencias = []

            # Verificar consistência de data/hora
            if 'EXIF DateTimeOriginal' in tags and 'EXIF DateTimeDigitized' in tags:
                if tags['EXIF DateTimeOriginal'] != tags['EXIF DateTimeDigitized']:
                    inconsistencias.append("Inconsistência nas datas")

            # Verificar informações da câmera
            if 'Image Make' not in tags or 'Image Model' not in tags:
                inconsistencias.append("Metadados da câmera incompletos")

            # Verificar configurações de exposição
            configuracoes_essenciais = ['EXIF ExposureTime', 'EXIF FNumber', 'EXIF ISOSpeedRatings']
            configs_presentes = sum(1 for config in configuracoes_essenciais if config in tags)
            if configs_presentes < 2:
                inconsistencias.append("Configurações de exposição ausentes")

            # Verificar se há metadados de software de edição
            software_tags = ['Software', 'Processing Software', 'History Software Agent']
            for software_tag in software_tags:
                if software_tag in tags:
                    inconsistencias.append(f"Software de edição detectado: {tags[software_tag]}")

            return {
                'total_inconsistencias': len(inconsistencias),
                'inconsistencias': inconsistencias,
                'manipulado': len(inconsistencias) > 1
            }

        except Exception as e:
            return {
                'total_inconsistencias': 1,
                'inconsistencias': [f"Erro na leitura: {str(e)}"],
                'manipulado': False
            }

    # 7. ANÁLISE DE RUÍDO DO SENSOR
    def analisar_ruido_sensor(self, imagem_path):
        """Analisa o padrão de ruído do sensor"""
        try:
            imagem = cv2.imread(imagem_path, cv2.IMREAD_GRAYSCALE)
            if imagem is None:
                return None

            # Calcular ruído residual (imagem suavizada - imagem original)
            imagem_suavizada = cv2.GaussianBlur(imagem, (5, 5), 0)
            ruido_residual = imagem.astype(float) - imagem_suavizada.astype(float)

            # Análise estatística do ruído
            variancia_ruido = np.var(ruido_residual)
            assimetria_ruido = np.mean((ruido_residual - np.mean(ruido_residual)) ** 3)
            if np.std(ruido_residual) ** 3 != 0:
                assimetria_ruido /= np.std(ruido_residual) ** 3
            else:
                assimetria_ruido = 0

            # Detectar inconsistências no ruído
            altura, largura = imagem.shape
            quadrantes = [
                ruido_residual[:altura // 2, :largura // 2],
                ruido_residual[:altura // 2, largura // 2:],
                ruido_residual[altura // 2:, :largura // 2],
                ruido_residual[altura // 2:, largura // 2:]
            ]

            variancias_quadrantes = [np.var(q) for q in quadrantes if q.size > 0]
            if len(variancias_quadrantes) > 0 and np.mean(variancias_quadrantes) != 0:
                inconsistencia_ruido = np.std(variancias_quadrantes) / np.mean(variancias_quadrantes)
            else:
                inconsistencia_ruido = 0

            return {
                'variancia_ruido': variancia_ruido,
                'assimetria_ruido': assimetria_ruido,
                'inconsistencia_ruido': inconsistencia_ruido,
                'manipulado': inconsistencia_ruido > 0.3
            }

        except Exception as e:
            print(f"   ⚠️ ERRO na análise de ruído: {e}")
            return {
                'variancia_ruido': 0,
                'assimetria_ruido': 0,
                'inconsistencia_ruido': 0,
                'manipulado': False
            }

    # 8. ANÁLISE COMPLETA COMBINADA
    def analise_completa(self, imagem_path):
        """Combina todas as técnicas para análise forense completa"""
        print("=" * 70)
        print("ANÁLISE FORENSE COMPLETA DE IMAGEM")
        print("=" * 70)

        resultados = {}
        votos_manipulacao = 0
        total_testes = 0

        # Lista de análises disponíveis
        analises = [
            ("Lei de Benford (DCT)", lambda x: self.analisar_lei_benford(x, 'dct')),
            ("Lei de Benford (Pixels)", lambda x: self.analisar_lei_benford(x, 'pixels')),
            ("Análise de Ruído do Sensor", self.analisar_ruido_sensor),
            ("Compressão Múltipla", self.detectar_compressao_multipla),
            ("Detecção de Clonagem", self.detectar_clonagem),
            ("Análise de Iluminação", self.analisar_iluminacao),
            ("Detecção de Resampling", self.detectar_resampling),
            ("Análise de Metadados", self.analisar_metadados)
        ]

        for nome, metodo in analises:
            try:
                print(f"\n🔍 {nome}:")
                resultado = metodo(imagem_path)
                if resultado is not None:
                    resultados[nome] = resultado

                    if resultado.get('manipulado', False):
                        votos_manipulacao += 1
                        print(f"   ❌ SUSPEITA - Possível manipulação detectada")
                    else:
                        print(f"   ✅ NORMAL - Sem indícios de manipulação")

                    total_testes += 1

                    # Mostrar métricas principais
                    for chave, valor in resultado.items():
                        if chave != 'manipulado' and not isinstance(valor, (list, dict)):
                            if isinstance(valor, float):
                                print(f"   {chave}: {valor:.4f}")
                            else:
                                print(f"   {chave}: {valor}")

            except Exception as e:
                print(f"   ⚠️ ERRO na análise: {str(e)}")

        # Plotar análise de Benford se disponível
        self._plotar_benford(resultados)

        # Resultado final
        print("\n" + "=" * 70)
        print("RESULTADO FINAL:")
        print("=" * 70)

        if total_testes > 0:
            confianca_manipulacao = (votos_manipulacao / total_testes) * 100
        else:
            confianca_manipulacao = 0

        if confianca_manipulacao > 70:
            print(f"❌ ALTA PROBABILIDADE DE MANIPULAÇÃO ({confianca_manipulacao:.1f}%)")
            print("   A imagem apresenta múltiplos indícios de alteração")
        elif confianca_manipulacao > 40:
            print(f"⚠️  SUSPEITA DE MANIPULAÇÃO ({confianca_manipulacao:.1f}%)")
            print("   Algumas análises indicam possível alteração")
        else:
            print(f"✅ IMAGEM PROVAVELMENTE AUTÊNTICA ({confianca_manipulacao:.1f}%)")
            print("   Poucos ou nenhum indício de manipulação detectado")

        return resultados

    def _plotar_benford(self, resultados):
        """Plota comparação da Lei de Benford se disponível"""
        benford_data = None
        for nome, resultado in resultados.items():
            if 'Lei de Benford' in nome and 'distribuicao_teorica' in resultado:
                benford_data = resultado
                break

        if benford_data:
            plt.figure(figsize=(10, 6))
            digitos = range(1, 10)
            plt.bar([d - 0.2 for d in digitos], benford_data['distribuicao_teorica'],
                    width=0.4, label='Teórico', alpha=0.7, color='blue')
            plt.bar([d + 0.2 for d in digitos], benford_data['distribuicao_observada'],
                    width=0.4, label='Observado', alpha=0.7, color='red')
            plt.xlabel('Primeiro Dígito')
            plt.ylabel('Frequência')
            plt.title('Lei de Benford - Distribuição dos Primeiros Dígitos')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()


# Função para selecionar imagem
def selecionar_imagem():
    root = tk.Tk()
    root.withdraw()

    arquivo = filedialog.askopenfilename(
        title="Selecione uma imagem para análise forense",
        filetypes=[
            ("Imagens", "*.jpg *.jpeg *.png *.bmp *.tiff"),
            ("Todos os arquivos", "*.*")
        ]
    )
    return arquivo


# Função principal
def main():
    print("=== ANALISADOR FORENSE DE IMAGENS ===")
    print("Este software analisa imagens para detectar manipulações e compressões múltiplas")
    print("Usando técnicas como Lei de Benford, análise de ruído, detecção de clonagem, etc.\n")

    # Selecionar imagem
    caminho_imagem = selecionar_imagem()

    if caminho_imagem and os.path.exists(caminho_imagem):
        print(f"📁 Imagem selecionada: {os.path.basename(caminho_imagem)}")
        print(f"📂 Caminho: {caminho_imagem}")
        print(f"📊 Tamanho: {os.path.getsize(caminho_imagem) / 1024 / 1024:.2f} MB")

        # Verificar se é uma imagem válida
        try:
            with Image.open(caminho_imagem) as img:
                print(f"🖼️  Dimensões: {img.size[0]} x {img.size[1]} pixels")
                print(f"📐 Formato: {img.format}")
        except:
            print("⚠️  Aviso: Não foi possível verificar os detalhes da imagem")

        print("\n" + "=" * 70)

        # Iniciar análise
        detector = DetectorManipulacaoAvancado()
        resultados = detector.analise_completa(caminho_imagem)

        print("\n" + "=" * 70)
        print("ANÁLISE CONCLUÍDA!")
        print("=" * 70)

        return resultados
    else:
        print("❌ Nenhuma imagem válida selecionada ou arquivo não encontrado.")
        return None


# Executar o programa
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  Análise interrompida pelo usuário")
    except Exception as e:
        print(f"\n\n💥 Erro crítico: {e}")
    finally:
        input("\nPressione Enter para sair...")

        