import os
import re
import difflib
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Inicializar o modelo CodeBERT
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModelForSequenceClassification.from_pretrained("microsoft/codebert-base")

# Função para normalizar o código
def normalizar_codigo(codigo):
    # Substituir nome da classe
    codigo = re.sub(r'public\s+class\s+\w+', 'public class CLASSNAME', codigo)
    # Substituir declarações de variáveis
    codigo = re.sub(r'(\b(int|String|double|float|char|boolean)\s+\w+)', '\1 VAR', codigo)
    # Substituir nomes de funções
    codigo = re.sub(r'(\b\w+\s+\w+\s*\()', 'FUNCNAME (', codigo)
    # Remover espaços em branco excessivos
    codigo = re.sub(r'\s+', ' ', codigo)
    # Remover comentários
    codigo = re.sub(r'//.*|/\*.*?\*/', '', codigo, flags=re.DOTALL)
    # Substituir números literais
    codigo = re.sub(r'\b\d+\b', 'NUMBER', codigo)
    return codigo

# Função para extrair operações chave
def extrair_operacoes(codigo):
    # Capturar estruturas chave, como loops, condicionais e chamadas de funções
    operacoes = []
    # Detectando loops e condicionais
    operacoes += re.findall(r'\b(for|while|if|else|switch|case)\b', codigo)
    # Detectando funções chamadas
    operacoes += re.findall(r'\b\w+\s*\(', codigo)
    return set(operacoes)

# Função para comparar a semântica do código
def comparar_semantica(codigo1, codigo2):
    # Extrair operações chave de ambos os códigos
    operacoes1 = extrair_operacoes(codigo1)
    operacoes2 = extrair_operacoes(codigo2)
    
    # Calcular a similaridade semântica com base nas operações encontradas
    similaridade_operacoes = len(operacoes1.intersection(operacoes2)) / max(len(operacoes1), len(operacoes2))
    return similaridade_operacoes

# Função para sugerir melhorias no código original quando a similaridade for baixa
def sugerir_melhorias(similarity_score):
    if similarity_score < 0.4:
        return "A similaridade entre os codigos esta baixa. Verifique os seguintes pontos no codigo original para melhorar a clareza e legibilidade:\n" \
               "- Considere usar nomes de variaveis mais descritivos.\n" \
               "- Tente modularizar o codigo, dividindo-o em funcoes menores e reutilizaveis.\n" \
               "- Verifique se a formatacao do codigo segue um estilo consistente (ex: indentacao e espacamento).\n" \
               "- Revise os tipos de dados usados para garantir precisao e clareza."
    else:
        return "A similaridade esta boa. O codigo parece estar em conformidade."

# Função para comparar as diferenças de código e retornar apenas os trechos com diferença
def comparar_diferencas(codigo1, codigo2):
    # Comparar as linhas dos dois códigos
    diff = difflib.ndiff(codigo1.splitlines(), codigo2.splitlines())
    
    # Filtrar apenas as diferenças (+ ou -)
    diferencas = [linha[2:] for linha in diff if linha.startswith('+ ') or linha.startswith('- ')]
    
    # Retornar as diferenças ou "Sem diferenças" caso não haja
    return '\n'.join(diferencas) if diferencas else "Sem diferencas."

# Criar a aplicação Flask
app = Flask(__name__)

@app.route('/')
def index():
    return "<h1>CodeBERT API - Similaridade de Código</h1><p>Envie requisições para /compare</p>", 200

@app.route('/compare', methods=['POST'])
def compare():
    try:
        # Obter os códigos do corpo da requisição
        data = request.get_json()
        codigo1 = data.get("codigo1")
        codigo2 = data.get("codigo2")

        if not codigo1 or not codigo2:
            return jsonify({"error": "Ambos os códigos devem ser fornecidos."}), 400

        # Calcular similaridade para os códigos originais
        inputs_original = tokenizer(codigo1, codigo2, return_tensors="pt", padding=True, truncation=True)
        outputs_original = model(**inputs_original)
        logits_original = outputs_original.logits
        similarity_original = torch.softmax(logits_original, dim=1)[0][1].item()

        # Normalizar os códigos
        codigo1_normalizado = normalizar_codigo(codigo1)
        codigo2_normalizado = normalizar_codigo(codigo2)

        # Calcular similaridade para os códigos normalizados
        inputs_normalizado = tokenizer(codigo1_normalizado, codigo2_normalizado, return_tensors="pt", padding=True, truncation=True)
        outputs_normalizado = model(**inputs_normalizado)
        logits_normalizado = outputs_normalizado.logits
        similarity_normalizado = torch.softmax(logits_normalizado, dim=1)[0][1].item()

        # Comparar semântica
        semantica_similarity = comparar_semantica(codigo1, codigo2)

        # Comparar as diferenças entre os códigos originais
        diferencas_codigo_original = comparar_diferencas(codigo1, codigo2)

        # Comparar as diferenças entre os códigos normalizados
        diferencas_codigo_normalizado = comparar_diferencas(codigo1_normalizado, codigo2_normalizado)

        # Ajuste na lógica de similaridade final
        if diferencas_codigo_original == "Sem diferencas." and diferencas_codigo_normalizado == "Sem diferencas.":
            similarity_final = 1.0  # Similaridade máxima caso não haja diferenças
        else:
            # Média ponderada das similaridades
            similarity_final = (0.6 * similarity_original) + (0.4 * similarity_normalizado) * (0.5 * semantica_similarity)

        # Sugerir melhorias caso a similaridade seja baixa
        melhorias = sugerir_melhorias(similarity_final)

        return jsonify({
            "similarity_original": similarity_original,
            "similarity_normalizado": similarity_normalizado,
            "semantica_similarity": semantica_similarity,
            "similarity_final": similarity_final,
            "diferencas_codigo_original": diferencas_codigo_original,
            #"diferencas_codigo_normalizado": diferencas_codigo_normalizado,
            "sugestoes_melhorias": melhorias
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
