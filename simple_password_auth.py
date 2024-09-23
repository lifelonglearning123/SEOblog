import streamlit as st

def creds_entered():
    if st.session_state["user"].strip() == "success" and st.session_state["passwd"].strip() == "moves":
        st.session_state["authenticated"] = True
    else:
        st.session_state["authenticated"] = False
        if not st.session_state["passwd"]:
            st.warning("Please enter password.")
        elif not st.session_state["user"]:
            st.warning("Please enter username.")
        else:
            st.error("Invalid Username/Password")

def authenticate_user():
	if "authenticated" not in st.session_state: 
		st.text_input(label="Username :", value = "", key = "user", on_change = creds_entered)
		st.text_input(label ="Password :", value = "", key = "passwd", type = "password", on_change = creds_entered)
		return False
	else: 
		if st.session_state["authenticated"]:
			return True
		else: 
			st.text_input(label="Username :", value = "", key="user", on_change = creds_entered)
			st.text_input(label="Password :", value = "", key="passwd", type = "password", on_change = creds_entered)
			return False

#if authenticate_user():  # If authenticated successfully, run the app