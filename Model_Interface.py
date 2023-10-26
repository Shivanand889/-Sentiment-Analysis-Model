from tkinter import *
from tkinter import messagebox as tmsg
from tensorflow.keras import models
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
root = Tk() 
root.geometry("600x500")
model = models.load_model("Sentiment_Model.model")
l1 = Label(text="Welcome to the Review System",font="TimesRoman 15 bold",bg="white")
l1.pack(fill ='x')
l2=Label(text="Choose The File Type for input",font="BookAntiqua 13 bold")
l2.pack(pady='34')
val= IntVar()
word= pd.read_csv("word_indexes.csv")
word_dict=dict(zip(word.Words ,word.Indexes))
def vec(text):
    arr=[]
    for w in text:
        if w in word_dict.keys():
            arr.append(word_dict[w])
    return arr

def review1():
    s = st.get().lower()
    x= s.split()
    x1=[]
    x1.append(vec(x))
    x1 = np.array(x1)
    x1 = pad_sequences(x1, maxlen=400, padding='post')
#     print(x)
    y= model.predict(x1)
    txt =""
    if(y>.5) :
        txt = "positive"
    else :
        txt = "Negative"
        
    ent2 = Label(text=f"Sentiment is {txt}").pack()
    ent2.pack(pady= 10)
    
def review2():
    data = pd.read_csv(st.get())
#     print(data.head())
    

    data1 = data.iloc[:,0]
    
    x = []
    for i in data1 :
        x.append(i.lower())
    data1 = data1.apply(lambda x : x.split()) 
    data1 = data1.apply(vec)
    data1 = pad_sequences(data1, maxlen=400, padding='post')
    y= model.predict(data1)
    y1 =[]
    
    
    for i in y :
        if(i>.5) :
             y1.append("Positive")
        else :
             y1.append("Negative")
                
    df = pd.DataFrame({"Review":x ,"Sentiment": y1})
    a= tmsg.showinfo("output", "Output of the file is saved in a file 'output.csv' in same folder")
    df.to_csv('output.csv')
st= StringVar()
def radio_val():
    x=val.get()
    
    if(x==1):
        ent = Label(text="Enter the review as a text").pack()
        ent1 = Entry(root,textvariable= st).pack()
        bt = Button(text="Get Sentiment",command=review1).pack(pady =10)
        
    else:
        ent = Label(text="Enter the Path of CSV file").pack()
        ent1 = Entry(root,textvariable= st).pack()
        bt = Button(text="Get Sentiment",command=review2).pack(pady =10)
        
        
entry = Radiobutton(text="Input Text",variable=val,value =1,command=radio_val).pack(anchor="nw" ,padx= 150)
entry = Radiobutton(text="CSV File",variable=val,value=2,command=radio_val).pack(anchor="nw" ,padx= 150)

root.mainloop()
