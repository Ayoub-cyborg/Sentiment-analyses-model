import streamlit as st
def main():
	st.title("Sentiment Analysis NLP App")
	st.subheader("Streamlit Projects")

	menu = ["Home","About"]
	choice = st.sidebar.selectbox("Menu",menu)

	if choice == "Home":
		st.subheader("Home")
		with st.form(key='nlpForm'):
			raw_text = st.text_area("Enter Text Here")
			submit_button = st.form_submit_button(label='Analyze')

		# layout
		col1,col2 = st.columns(2)
		if submit_button:

			with col1:
				pass

				



			with col2:
				st.info("Token Sentiment")

				pass





	else:
		st.subheader("About")


if __name__ == '__main__':
	main()
