import tkinter as tk
from tkinter import filedialog
import os


def selecionar_imagem_interativo():
    """Abre uma janela para selecionar a imagem interativamente"""
    root = tk.Tk()
    root.withdraw()  # Esconde a janela principal

    # Abre o diálogo de arquivo começando na pasta de downloads
    caminho_downloads = os.path.join(os.path.expanduser('~'), 'Downloads')

    arquivo = filedialog.askopenfilename(
        initialdir=caminho_downloads,
        title="Selecione uma imagem",
        filetypes=[
            ("Imagens", "*.jpg *.jpeg *.png *.bmp *.tiff"),
            ("JPEG", "*.jpg *.jpeg"),
            ("PNG", "*.png"),
            ("Todos os arquivos", "*.*")
        ]
    )
    return arquivo


# Usar o método interativo
caminho_imagem = selecionar_imagem_interativo()

if caminho_imagem:
    print(f"Imagem selecionada: {caminho_imagem}")