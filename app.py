from flask import *
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


app=Flask(__name__)



#def build_model():
    #vec=pickle.load(open("tfidfvect.pkl","rb"))
    #df=pickle.load(open("df.pkl","rb"))
    #d=df[["clean_text1","clean_text2"]]
    #vectors = vec.transform(d)
    #cosine_similarities = cosine_similarity(vectors)
    #model={"cosine_similarities": cosine_similarities, "vectorizer":vec, "vectors":vectors}
    #return model

def check_score(text1,text2):
    #df=pickle.load(open("df.pkl","rb"))
    vec=pickle.load(open("tfidfvect.pkl","rb"))
    x=vec.fit_transform([text1])
    y=vec.transform([text2])
    #vectors=model["vectors"]
    s_score=cosine_similarity(x,y)[0]
    score=s_score.max()
    return score
    
    
    
@app.route("/")
def homepage():
    return render_template("home.html")

@app.route("/senddata",methods=["POST"])
def fetchdata():
    #model=build_model()

    text1=request.form["t1"]
    text2=request.form["t2"]
    
    score=check_score(text1,text2)
    return render_template("display.html",data=score)


if (__name__=="__main__"):
    app.run(debug=True)