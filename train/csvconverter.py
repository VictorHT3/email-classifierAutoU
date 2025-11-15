import csv
import random
import os

caminho = r"C:\Users\vitin\PycharmProjects\Email_Classifier_AutoU\train\sample_data.csv"

os.makedirs(os.path.dirname(caminho), exist_ok=True)

produtivo = [
    "Preciso de uma atualização sobre o chamado {}.",
    "Anexo o relatório mensal. Favor confirmar recebimento.",
    "Existe alguma previsão para a entrega do documento?",
    "Solicito a correção do boleto em anexo.",
    "Pode me enviar os arquivos do projeto?",
    "Reunião marcada para amanhã às 10h.",
    "Favor revisar o contrato antes da assinatura.",
    "Atualizei a planilha de custos.",
    "Confirmei a disponibilidade do fornecedor.",
    "Encaminhei o orçamento solicitado.",
    "Atualização de status do projeto: {}% concluído.",
    "Preciso que revisem o documento até sexta-feira.",
    "Agendei a reunião com o cliente para segunda.",
    "Favor enviar o relatório financeiro atualizado.",
    "Confirmei a presença de todos na reunião.",
    "Solicito ajustes na apresentação para a diretoria.",
    "Atualizei os indicadores de desempenho.",
    "Encaminhei a ata da reunião para aprovação.",
    "Preciso que finalizem os relatórios até o fim do dia.",
    "Favor validar os dados do sistema.",
    "Enviei o plano de ação solicitado.",
    "Atualizei os tickets do chamado {}.",
    "Solicito aprovação do orçamento urgente.",
    "Favor revisar o código antes do deploy.",
    "Confirmei o envio das notas fiscais."
]

improdutivo = [
    "Bom dia! Feliz Natal a todos.",
    "Agradeço pelo suporte, tudo certo por aqui.",
    "Parabéns pelo trabalho realizado!",
    "Oi, tudo bem? Como foi seu final de semana?",
    "Vou passar no café com vocês mais tarde.",
    "Boa tarde, pessoal! Vamos almoçar juntos?",
    "Parabéns pelo aniversário!",
    "Como estão todos por aí?",
    "Excelente trabalho no evento de ontem!",
    "Que foto linda você postou!",
    "Feliz Ano Novo a todos!",
    "Que legal o vídeo que você enviou!",
    "Obrigado pelo café!",
    "Boa noite, pessoal!",
    "Parabéns pela promoção!",
    "Que gostoso o almoço hoje!",
    "Que música ótima você compartilhou!",
    "Bom final de semana a todos!",
    "Obrigado pelo presente!",
    "Oi, vamos nos encontrar amanhã?"
]

with open(caminho, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["text", "label"])
    for i in range(250):
        writer.writerow([random.choice(produtivo).format(random.randint(100,9999)), "Produtivo"])
        writer.writerow([random.choice(improdutivo), "Improdutivo"])

print(f"Arquivo CSV salvo com sucesso em: {caminho}")