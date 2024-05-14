import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier , plot_tree
df = pd.read_csv('dataset.csv')
df.drop_duplicates(inplace=True)
df["sex"].value_counts().plot(kind = "pie", figsize= (6,6), shadow = True)
df["age"].value_counts().plot(kind = "bar", figsize= (30,5))
# Correlation Matrix
corrmat = df.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(15,15))
g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")
sns.set_style('whitegrid')
sns.countplot(x='target',data=df,palette='RdBu_r')
label = df['target']
feature = df.drop(['target'], axis = 1)
feature = pd.get_dummies(feature)

corr_matrix = feature.corr()

sns.heatmap(corr_matrix, cmap='coolwarm')

plt.title('Correlation Heatmap')

plt.show()
for col in feature.columns:
    q1, q3 = np.percentile(feature[col], [25, 75])
    iqr = q3 - q1
    threshold = 1.5 * iqr
    outliers = (feature[col] < q1 - threshold) | (feature[col] > q3 + threshold)
    num_outliers = np.sum(outliers)
    print(f"Outliers No. in {col}: {num_outliers}")
f_encoded = pd.get_dummies(feature)
l_encoded = pd.get_dummies(label)
X= f_encoded
y= l_encoded
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
dtc = DecisionTreeClassifier()
dtc.fit(x_train, y_train)
y_pred = dtc.predict(x_test)
plt.figure (figsize = (10,10))
plot_tree (dtc , filled = True)
plt.show ()
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR

# Assuming X is your feature matrix and y is your target matrix
svr = SVR()
multi_output_svr = MultiOutputRegressor(svr)
multi_output_svr.fit(x_train, y_train)
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LogisticRegression

# Assuming X is your feature matrix and y is your target matrix
reg = LogisticRegression()
LG = MultiOutputRegressor(reg)
LG.fit(x_train, y_train)
models = {
    'Disicion tree': dtc,
    'SVC_model': multi_output_svr,
    'logistic':LG
}

# Initialize a dictionary to store accuracies
accuracies = {}
for name, model in models.items():
    # Make predictions on the testing set
    y_pred = model.predict(x_test)
    # Round the predicted values to the nearest integer (if needed)
    y_pred = np.round(y_pred)
    # Calculate accuracy and store it
    y_pred_int = y_pred.astype(int)
    acc = accuracy_score(y_test, y_pred_int)
    accuracies[name] = acc
# Print accuracies
for name, acc in accuracies.items():
    print(f'{name} Accuracy:', acc)


from tkinter import  *
from tkinter import  messagebox

root = Tk()
# root.maxsize(800,800)
root.configure(width="600",height="600",bg="lightblue")
# root.minsize(200,200)
root.title("heart prediction")
label2 = Label(root , text="Enter age")
label2.configure(bg="blue" , foreground="white" , font=("Arial" ,15 , "bold"))
label2.pack()
age = Entry(root, width=40 , foreground="white" , bg = "gray")
age.pack(pady=10)
label3 = Label(root , text="Enter sex (1 for male and 0 for female) ")
label3.configure(bg="blue" , foreground="white" , font=("Arial" ,15 , "bold"))
label3.pack()
sex = Entry(root, width=40 , foreground="white" , bg = "gray")
sex.pack(pady=10)
label4 = Label(root , text="Enter cp ")
label4.configure(bg="blue" , foreground="white" , font=("Arial" ,15 , "bold"))
label4.pack()
cp = Entry(root, width=40 , foreground="white" , bg = "gray")
cp.pack(pady=10)
label5 = Label(root , text="Enter trestbps ")
label5.configure(bg="blue" , foreground="white" , font=("Arial" ,15 , "bold"))
label5.pack()
trestbps = Entry(root, width=40 , foreground="white" , bg = "gray")
trestbps.pack(pady=10)
label6 = Label(root , text="Enter cholestrol ")
label6.configure(bg="blue" , foreground="white" , font=("Arial" ,15 , "bold"))
label6.pack()
chol = Entry(root, width=40 , foreground="white" , bg = "gray")
chol.pack(pady=10)

symptoms = {"age":0,"sex":0,"cp":0,"trestbps":0,"cholestrol":0,"fbs":0,"restecg":0,"thalach":0,"exang":0,"oldpeak":0,"slope":0,"ca":0,"thal":0}
def submit():
    symptoms["age"] = float(age.get())
    symptoms["sex"] = float(sex.get())
    symptoms["cp"] = float(cp.get())
    symptoms["trestbps"] = float(trestbps.get())
    symptoms["cholestrol"] = float(chol.get())
    symptoms["fbs"] = float(fbs.get())
    symptoms["restecg"] = float(restecg.get())
    symptoms["thalach"] = float(thalach.get())
    symptoms["exang"] = float(exang.get())
    symptoms["oldpeak"] = float(oldpeak.get())
    symptoms["slope"] = float(slope.get())
    symptoms["ca"] = float(ca.get())
    symptoms["thal"] = float(thal.get())
    # Reshape the data array
    data_list = [symptoms[key] for key in symptoms.keys()]
    data_array = np.array(data_list).reshape(1, -1)

    pred = LG.predict(data_array)
    res = "you have heart disease" if pred[0][0] == 1 else "you don't have heart disease"
    messagebox.showinfo("Detection", res)


label7 = Label(root, text="Enter fbs")
label7.configure(bg="blue", foreground="white", font=("Arial", 15, "bold"))
label7.pack()
fbs = Entry(root, width=40, foreground="white", bg="gray")
fbs.pack(pady=5)

label8 = Label(root, text="Enter restecg")
label8.configure(bg="blue", foreground="white", font=("Arial", 15, "bold"))
label8.pack()
restecg = Entry(root, width=40, foreground="white", bg="gray")
restecg.pack(pady=5)

label9 = Label(root, text="Enter thalach")
label9.configure(bg="blue", foreground="white", font=("Arial", 15, "bold"))
label9.pack()
thalach = Entry(root, width=40, foreground="white", bg="gray")
thalach.pack(pady=5)

label10 = Label(root, text="Enter exang")
label10.configure(bg="blue", foreground="white", font=("Arial", 15, "bold"))
label10.pack()
exang = Entry(root, width=40, foreground="white", bg="gray")
exang.pack(pady=5)

label11 = Label(root, text="Enter oldpeak")
label11.configure(bg="blue", foreground="white", font=("Arial", 15, "bold"))
label11.pack()
oldpeak = Entry(root, width=40, foreground="white", bg="gray")
oldpeak.pack(pady=5)

label12 = Label(root, text="Enter slope")
label12.configure(bg="blue", foreground="white", font=("Arial", 15, "bold"))
label12.pack()
slope = Entry(root, width=40, foreground="white", bg="gray")
slope.pack(pady=5)

label13 = Label(root, text="Enter ca")
label13.configure(bg="blue", foreground="white", font=("Arial", 15, "bold"))
label13.pack()
ca = Entry(root, width=40, foreground="white", bg="gray")
ca.pack(pady=5)

label14 = Label(root, text="Enter thal")
label14.configure(bg="blue", foreground="white", font=("Arial", 15, "bold"))
label14.pack()
thal = Entry(root, width=40, foreground="white", bg="gray")
thal.pack(pady=5)

submit_button = Button(root, text="Submit", command=submit)
submit_button.pack(pady=10)
root.mainloop()