from transformers import pipeline 
import tokenizers
import streamlit as st 
import copy


@st.cache(hash_funcs={tokenizers.Tokenizer: lambda _: None, tokenizers.AddedToken: lambda _: None})
def get_model() :
    return pipeline("sentiment-analysis", model='akhooli/xlm-r-large-arabic-sent')

input = st.text_input('Text')
bt = st.button("Get Sentiment Analysis")

if bt and input:
    model = copy.deepcopy(get_model())
    st.write(model(input))