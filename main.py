
import streamlit as st
from few_shot import fewShotPosts
from post_generator import generate_post



language_options =["English","German","French","Hinglish"]

def main():
    st.title("LinkedIN Post Generator")
    col1,col2,col3 = st.columns(3)
    fs = fewShotPosts()
    with col1:
        selected_tag = st.selectbox("Topics",options=fs.get_tags())
    with col2:
        selected_length = st.selectbox("Length",options=["Short","Medium","Long"])
    with col3:
        selected_language = st.selectbox("Language",options=language_options)
    if st.button("Generate"):
        post = generate_post(selected_length,selected_language,selected_tag)
        st.write(post)

if __name__ == "__main__":
    main()


