import streamlit_authenticator as stauth

passwords = ["admin"]
hashed_passwords = stauth.Hasher(passwords).generate()
print(hashed_passwords)