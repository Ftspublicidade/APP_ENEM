import streamlit as st
import pandas as pd
from joblib import dump, load

regr_carregado = load('modelo_enem.joblib') 

st.title("Como a situação socioecoômica pode interferir na Matemática do ENEM?")

st.markdown("Para fomentar o debate sobre fatores que influenciam no desempenho das\
        acadêmico dos alunos foi treinado um modelo de Aprendizagem de Máquina\
        utilizando as 8 variáveis mais significativas para predizer a nota de \
        Matemática de uma pessoa ao realizar a nota do ENEM.")

st.markdown("O modelo foi desenvolvido para FINS EDUCACIONAIS e não deve ser levado\
        em consideração para qualquer outro intuito que não seja fomentar o \
        debate e gerar discussões produtivas")

st.subheader("Responda as seguintes perguntas para estimar uma nota do ENEM")

q002_opcoes = {'Nunca estudou.':'A',
        'Não completou a 4ª série/5º ano do Ensino Fundamental.':'B',
        'Completou a 4ª série/5º ano, mas não completou a 8ª série/9º ano do Ensino Fundamental.':'C',
        'Completou a 8ª série/9º ano do Ensino Fundamental, mas não completou o Ensino Médio.':'D',
        'Completou o Ensino Médio, mas não completou a Faculdade.':'E',
        'Completou a Faculdade, mas não completou a Pós-graduação.':'F',
        'Completou a Pós-graduação.':'G',
        'Não sei.':'H'}

q002 = st.selectbox(
        'Até que série sua mãe, ou a mulher responsável por você, estudou?',
        options=list(q002_opcoes.keys()))

quantidade_opcoes = {'Não.':'A',
        'Sim, um.':'B',
        'Sim, dois.':'C',
        'Sim, três.':'D',
        'Sim, quatro ou mais.':'E'}

q008 = st.selectbox(
        'Na sua residência tem banheiro?',
        options=list(quantidade_opcoes.keys()))

q010 = st.selectbox(
        'Na sua residência tem carro?',
        options=list(quantidade_opcoes.keys()))

sim_nao_opcoes = {'Não.':'A',
        'Sim.':'B'}

q018 = st.selectbox(
        'Na sua residência tem aspirador de pó?',
        options=list(sim_nao_opcoes.keys()))

q019 = st.selectbox(
        'Na sua residência tem televisão em cores?',
        options=list(quantidade_opcoes.keys()))

q022 = st.selectbox(
        'Na sua residência tem telefone celular?',
        options=list(quantidade_opcoes.keys()))

q024 = st.selectbox(
        'Na sua residência tem computador?',
        options=list(quantidade_opcoes.keys()))

q027_opcoes = {'Somente em escola pública.':'A',
        'Parte em escola pública e parte em escola privada SEM bolsa de estudo integral.':'B',
        'Parte em escola pública e parte em escola privada COM bolsa de estudo integral.':'C',
        'Somente em escola privada SEM bolsa de estudo integral.':'D',
        'Somente em escola privada COM bolsa de estudo integral.':'E',
        'Não frequentei a escola':'F'}

q027 = st.selectbox(
        'Em que tipo de escola você frequentou o Ensino Médio?',
        options=list(q027_opcoes.keys()))

if st.button('Calcular nota'):
    df_predict = pd.DataFrame()
    
    df_predict['Q002_CAT'] = pd.Categorical(q002_opcoes[q002], categories=['A','B','C','D','E','F','G','H'], ordered=True).codes 
    df_predict['Q008_CAT'] = pd.Categorical(quantidade_opcoes[q008], categories=['A','B','C','D','E','F','G','H'], ordered=True).codes
    df_predict['Q010_CAT'] = pd.Categorical(quantidade_opcoes[q010], categories=['A','B','C','D','E','F','G','H'], ordered=True).codes
    df_predict['Q018_CAT'] = pd.Categorical(sim_nao_opcoes[q018], categories=['A','B','C','D','E','F','G','H'], ordered=True).codes 
    df_predict['Q019_CAT'] = pd.Categorical(quantidade_opcoes[q019], categories=['A','B','C','D','E','F','G','H'], ordered=True).codes 
    df_predict['Q022_CAT'] = pd.Categorical(quantidade_opcoes[q022], categories=['A','B','C','D','E','F','G','H'], ordered=True).codes 
    df_predict['Q024_CAT'] = pd.Categorical(quantidade_opcoes[q024], categories=['A','B','C','D','E','F','G','H'], ordered=True).codes 
    df_predict['Q027_CAT'] = pd.Categorical(q027_opcoes[q027], categories=['A','B','C','D','E','F','G','H'], ordered=True).codes 
    
    nota = regr_carregado.predict(df_predict)
    
    st.text("Nota calculada pelo modelo de predição: "+str(nota[0]))
    
