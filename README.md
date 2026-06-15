# Classificação de Contatos Acústicos Subaquáticos com Redes Neurais Profundas

Repositório do artigo submetido ao **SIGE 2026 — ITA**.

---

## 🚀 Arquiteturas Avaliadas
- CNN Baseline
- CRNN
- Transformer-DWC (Wang et al., 2024)
- Conformer (Gulati et al., 2020)

---

## 📊 Dataset
- IARA (Silva et al., 2025)
- ShipsEar (Santos-Domínguez et al., 2016)

---

## 🏆 Resultados Principais

| Modelo | ACC (%) | Std (%) |
| :--- | :---: | :---: |
| CRNN | 66,14 | ±1,17 |
| Conformer | 65,23 | ±0,53 |
| DWC | 63,83 | ±1,04 |
| CNN | 63,18 | ±1,24 |

---

## 🛠️ Como Usar e Citação

```bash
# 1. Instalar as dependências:
pip install -r requirements.txt

# 2. Executar o treinamento:
python pipeline/train.py

@inproceedings{kfuri2026,
  author    = {Kfuri de Oliveira, Leandro and Fernandes, Rigel Procópio},
  title     = {Classificação de Contatos Acústicos Subaquáticos Utilizando Redes Neurais Profundas},
  booktitle = {SIGE 2026},
  year      = {2026}
}
