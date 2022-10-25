import streamlit as st

st.set_page_config(
    page_title="Question Answering Robustness",
    layout="centered",
)

st.markdown("## Exploring The Landscape of Distributional Robustness for Question Answering Model")
st.markdown("[Anas Awadalla](), [Mitchell Wortsman](), [Gabriel Ilharco](), [Sewon Min](), [Ian Magneusson](), [Hannaneh Hajishirzi](), [Ludwig Schmidt]()")

links = '[📄 Paper](https://arxiv.org/abs/2210.12517) \ [👨🏽‍💻 Code](https://github.com/allenai/catwalk)'
st.markdown(links)

st.markdown("### Abstract")
st.markdown("""We conduct a large empirical evaluation to
investigate the landscape of distributional 
robustness in question answering. Our 
investigation spans over 350 models and 16
question answering datasets, including a 
diverse set of architectures, model sizes, and
adaptation methods (e.g., fine-tuning, adapter
tuning, in-context learning, etc.). We find
that, in many cases, model variations do
not affect robustness and in-distribution 
performance alone determines out-of-distribution
performance. Moreover, our findings indicate
that i) zero-shot and in-context learning methods
are more robust to distribution shifts than
fully fine-tuned models; ii) few-shot prompt
fine-tuned models exhibit better robustness
than few-shot fine-tuned span prediction models;
iii) parameter-efficient and robustness 
enhancing training methods provide no significant
robustness improvements. In addition,
we publicly release all evaluations to encourage
researchers to further analyze robustness
trends for question answering models""")

st.markdown("### Findings")
st.image("overview.png", use_column_width=True)
st.markdown("""
##### Other conclusions:
❌ Model size is not correlated with robustness\\
🏛️ Architecture does not impact robustness\\
📉 As the number of training examples increases, robustness decreases
""")

st.markdown("Use the 'Data' tab to further explore our data.")

st.text("Please refer to our paper for more discussion on our findings.")

st.markdown("### Citation")
st.code("""
        @article{awadalla2022exploring,
        title={Exploring the Landscape of Distributional Robustness for Question Answering Models},
        author={Awadalla, Anas and Wortsman, Mitchell and Ilharco, Gabriel and Min, Sewon and Magneusson, Ian and Hajishirzi, Hannaneh and Schmidt, Ludwig},
        journal={EMNLP Findings 2022},
        year={2022}
        url={https://arxiv.org/abs/2210.12517}
        }""", language="text")