import csv
import random
import os

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "sample_data.csv")
os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)

produtivo = [
    "Preciso de uma atualização sobre o chamado {}.",
    "Anexo o relatório mensal. Favor confirmar recebimento.",
    "Existe alguma previsão para a entrega do documento?",
    "Solicito a correção do boleto em anexo.",
    "Pode me enviar os arquivos do projeto?",
    "Reunião marcada para amanhã às 10h.",
]

improdutivo = [
    "Bom dia! Feliz Natal a todos.",
    "Agradeço pelo suporte, tudo certo por aqui.",
    "Parabéns pelo trabalho realizado!",
    "Oi, tudo bem? Como foi seu final de semana?",
]

with open(DATA_PATH, mode="w", newline="", encoding="utf-8-sig") as file:
    writer = csv.DictWriter(file, fieldnames=["text", "label"], quoting=csv.QUOTE_MINIMAL)
    writer.writeheader()
    for i in range(250):
        writer.writerow({"text": random.choice(produtivo).format(random.randint(100,9999)), "label": "Produtivo"})
        writer.writerow({"text": random.choice(improdutivo), "label": "Improdutivo"})

print(f"CSV seguro gerado em: {DATA_PATH}")
