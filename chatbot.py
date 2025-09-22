from flask import Flask,request,jsonify
import mysql.connector
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app=Flask(__name__)

def get_connection ():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="123456",
        database="chatbot_db"
    )
faq={
    "thoi khoa bieu":"Ban co the xem thoi khoa bieu tai bang shedule o trang sinh vien",
    "mon hoc":"Cac mon hoc dang mo:Toan roi rac,Cau truc du lieu ,Machine learning,Deep Learning,Python",
    "Diem danh":"Diem danh duoc luu trong he thong quan ly lop hoc"
}
vectorizer=TfidfVectorizer()
faq_questions=list(faq.keys())
x=vectorizer.fit_transform(faq_questions)

def chatbot_response (user_input ):
    user_vec=vectorizer.transform([user_input])
    similarity=cosine_similarity(user_vec,x)
    idx=similarity.argmax()
    if similarity[0,idx]>0.3:
        return faq[faq_questions[idx]]
    else :
        return "xin loi,minh chua hieu cau hoi nay"
@app.route ("/chat",methods=["Post"])
def chat ():
    data=request.json
    user_input =data.get("message","")
    resonse=chatbot_response(user_input)
    return jsonify({"reply":resonse})
if __name__=="__main__":
    app.run (debug=True )    
    