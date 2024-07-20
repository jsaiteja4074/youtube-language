from flask import Flask, request, render_template
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import joblib
app=Flask(__name__)
# Load the trained model and vectorizer
model = joblib.load('model.pkl')
cv = joblib.load('bow_model.pkl')
tv = joblib.load('tf_idf_model.pkl')

app = Flask(__name__,template_folder='template')

# Define the home page route
@app.route('/')
def home():
    return render_template('index.html')

#@app.route('/predict')
# pred():
 #   return render_template('index.html')

# Define the prediction function
@app.route('/predict', methods=['POST'])
def predict():
    # Get the user input from the form
    description = request.form['description']
    if not description or description.strip() == '':
            # Return an error message if the input is empty
            return render_template('error.html', message='Please enter a description')

    if description is None or not isinstance(description, str):
        print(f"Invalid input: {type(description)}, {description}")
        return ""
    
    print(f"Valid input: {type(description)}, {description}")

    # print the type and value of description
    
    #Removing link
    url_pattern = r'((http|ftp|https):\/\/)?[\w\-_]+(\.[\w\-_]+)+([\w\-\.,@?^=%&amp;:/~\+#]*[\w\-\@?^=%&amp;/~\+#])?'
    description = re.sub(url_pattern, ' ', description)
    print(f"After removing links: {description}")

    #removing email address
    email_pattern =r'[a-z0-9\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z]+'
    description = re.sub(email_pattern, ' ', description)
    print(f"After removing email address: {description}")

    # replace every special char with space
    description = re.sub('[^a-zA-Z0-9\n]', ' ', description)
    print(f"After removing special characters: {description}")

    # replace multiple spaces with single space
    description = re.sub('\s+',' ', description)
    print(f"After removing extra spaces: {description}")

    # converting all the chars into lower-case.
    description = description.lower()
    print(f"After converting to lower-case: {description}")

    # stemming the description
    try:
        ps = PorterStemmer()
        description = ' '.join([ps.stem(word) for word in description.split() if not word in set(stopwords.words('english'))])
        print(f"After stemming: {description}")
    except Exception as e:
        print(f"Stemming error: {e}")
        return ""
    
    # Vectorize the input using the trained vectorizer
    X = tv.transform([description])
    
    # Make the prediction using the trained model
    result = model.predict(X)[0]
    
    # Return the predicted category to the user
    return render_template('result.html', prediction_text='Video category prediction Result: {}'.format(result))



if __name__ == '__main__':

    # Run the Flask app
    app.run(debug=True)
