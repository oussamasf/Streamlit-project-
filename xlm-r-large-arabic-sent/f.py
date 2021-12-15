from transformers import pipeline 
import streamlit as st 
import tokenizers
import copy
import matplotlib.pyplot as plt

def plot(score, label) : 
    labels = label, '___'
    sizes = [score,1-score]
    explode = (0, 0.1) # only "explode" the 2nd slice (i.e. 'Hogs')

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    return st.pyplot(fig1)


#Add title and subtitle to the main interface of the app
st.title("الإنطباعات")

#Add sidebar to the app
st.sidebar.markdown("### تطبيق تحليل النصوص")
st.sidebar.markdown("هذا تطبيق يستعمل المحولات اللغوية لتحليل النصوص و تحديد الإنطباعات الايجابية ، السلبية او المحايدة")

@st.cache(hash_funcs={tokenizers.Tokenizer: lambda _: None, tokenizers.AddedToken: lambda _: None})
def get_model() :
    return pipeline("sentiment-analysis", model='akhooli/xlm-r-large-arabic-sent')


st.markdown("<h6 style='text-align: right; color: grey;'>النص</h6>", unsafe_allow_html=True)
input = st.text_input(' ')
bt = st.button("تحليل النص")

if bt and input:
    # 0 mixed, 1 negative, 2 positive

    model = copy.deepcopy(get_model())
    a = model(input)
    if a[0]['label'] == 'LABEL_0' : 
        st.write("تظهر نتائج ان النص محايد بنسبة ")
        plot( a[0]['score'] , 'neutral')
    elif a[0]['label'] == 'LABEL_1' : 
        st.write("تظهر نتائج ان النص سلبي بنسبة ")
        plot( a[0]['score'] , 'negative')
    else : 
        st.write("تظهر نتائج ان النص إيجابي بنسبة ")
        plot( a[0]['score'] , 'positive')





# from transformers import pipeline 
# import streamlit as st 


# @st.cache(allow_output_mutation=True)
# def model(a="sentiment-analysis" , b='akhooli/xlm-r-large-arabic-sent') :
#     return pipeline(a, model=b)



# text = st.text_area("Enter some text in arabic language!") 
# if text: 
#   st.write(model()(text))

