# import necessary libraries
from flask import Flask, render_template, request, redirect, url_for, flash,session, jsonify
from flask_sqlalchemy import SQLAlchemy
#deteting perpose
#import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
# For working with regular expressions (pattern matching in text)
# For interacting with the operating system (files, folders, paths)
# For reading and writing CSV files
# For handling input/output streams in memory (file-like objects)
import re
import os
import csv
import io
#remove stop words from word cloud 
from wordcloud import STOPWORDS
# to show error if youtube api fails
from googleapiclient.errors import HttpError
# imported to create a service object its act as a client for interacting with google api
from googleapiclient.discovery import build
# to save and serve word cloud images
from flask import send_file
#imported to generate word cloud for youtube comments 
import matplotlib
matplotlib.use('Agg')   # Important for Flask + PDF
from wordcloud import WordCloud
import matplotlib.pyplot as plt
#api key for youtube data api 
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")
# to handle data and time for prediction history table 
from datetime import datetime
#import model related libararies 
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
#use built-in neural network functions like softmax,sigmoid ect 
import torch.nn.functional as F
#creating flask app and secret key
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "sentimodel_key_for_session")

# Database  configaration
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///users.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
#intializing database
db = SQLAlchemy(app)
#load model and set path
model_name = "uv1234/sentiment-analysis-transformer"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
# device config if gpu is avaliable then use it otherwise use cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
# set neutral threshold set 75%
NEUTRAL_THRESHOLD = 0.75
# Database Model(user table) this create database table with id,username,email and password 
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
# Prediction History Table( this table stores the history of predictions mode by user)
class PredictionHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False)
    date_time = db.Column(db.String(50))
    type = db.Column(db.String(20))  # text, file or youtube comments
    input_text = db.Column(db.Text)
    sentiment = db.Column(db.String(20))
    confidence = db.Column(db.Float)
    polarity = db.Column(db.Float)
    file_name = db.Column(db.String(100))
    positive_count = db.Column(db.Integer)
    negative_count = db.Column(db.Integer)
    neutral_count = db.Column(db.Integer)
    conf_90_100 = db.Column(db.Integer)
    conf_80_90 = db.Column(db.Integer) 
    conf_70_80 = db.Column(db.Integer)


# Create DB automatically,Creates users.db,Creates "user" table if not exists
with app.app_context():
    db.create_all()


#(B1) This is your main prediction code, and it is reused everywhere according to requirements
def predict_sentiment(text):

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
    # apply softmax to get probabilities 
    probabilities = F.softmax(outputs.logits, dim=1)
    negative_score = probabilities[0][0].item()
    positive_score = probabilities[0][1].item()
    predicted_class = torch.argmax(probabilities).item()
    confidence = torch.max(probabilities).item()
    # polarity score (-1 to +1)
    polarity_score = round(positive_score - negative_score, 3)
    confidence_percent = round(confidence * 100, 2)

    if confidence < NEUTRAL_THRESHOLD:

        sentiment = "Neutral"
        color = "yellow"
        icon = "fa-meh"

    elif predicted_class == 1:

        sentiment = "Positive"
        color = "green"
        icon = "fa-smile"

    else:

        sentiment = "Negative"
        color = "red"
        icon = "fa-frown"
    # return statement 
    return sentiment, confidence_percent, color, icon, polarity_score  

#(B2) Home Routes
@app.route("/")
def home():
    return render_template("home.html")




#(B3) Registration route GET=>SHOWS FORM ,POST=>process form data,getting form data 
@app.route("/register", methods=["GET", "POST"])
def register(): #Reads user input from HTML form
    if request.method == "POST":
        username = request.form["username"]
        email = request.form["email"].lower()
        password = request.form["password"]
        confirm = request.form["confirm"]

        # Username Validation
        if len(username) < 3 or username.isdigit() or not re.match("^[A-Za-z0-9]+$", username):
            flash("Username must contain at least 3 characters", "error")
            return redirect(url_for("register"))

        #  Email Validation 
        if not re.match(r"^[a-zA-Z0-9._%+-]+@gmail\.com$", email):
            flash("Email must end with @gmail.com", "error")
            return redirect(url_for("register"))

        # Password Validation 
        if len(password) < 8:
            flash("Password must contain at least 8 characters", "error")
            return redirect(url_for("register"))

        if password != confirm: #Prevents user mistakes
            flash("Passwords do not match", "error")
            return redirect(url_for("register"))

        #  Check Existing User 
        existing_user = User.query.filter(
            (User.username == username) | (User.email == email)
        ).first()

        if existing_user: #revents duplicate accounts
            flash("This user already exists, try another one", "error")
            return redirect(url_for("register"))

        # Hash Password 
        hashed_password = generate_password_hash(password)

        #Save User
        new_user = User(
            username=username,
            email=email,
            password=hashed_password
        )
         #saving user to database
        db.session.add(new_user)
        db.session.commit()

        return redirect(url_for("login"))

    return render_template("register.html")

#(B4) login route 
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        login_input = request.form["login"].lower()#Login Input 
        password = request.form["password"]

        user = User.query.filter(
            (User.username == login_input) | (User.email == login_input)
        ).first()
         #authentication logic

        if user and check_password_hash(user.password, password):
            session["user_id"]=user.id
            session["username"]=user.username
            session["email"] = user.email
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid username or password", "error")
            return redirect(url_for("login"))

    return render_template("login.html")

#(B5)dashboard route
@app.route("/dashboard")
def dashboard():
    if "user_id" not in session:
        flash("Please login first","error")
        return redirect(url_for("login"))
    return render_template("dashboard.html")



#(B6) History route
@app.route("/history")
def history():
    if "user_id" not in session:
        flash("Please login first", "error")
        return redirect(url_for("login"))
    #Fetch history records from database according to the user id 
    records = PredictionHistory.query.filter_by(
        user_id=session["user_id"]
    ).order_by(PredictionHistory.id.desc()).all()

    return render_template("history.html", records=records)



# (B7)history vew route, shows details of a specific prediction record based on record_id, checks if user is logged in and if record belongs to the user before displaying details
@app.route("/history/view/<int:record_id>")
def view_history(record_id):

    if "user_id" not in session:
        flash("Please login first", "error")
        return redirect(url_for("login"))
    #Fetch specific record from database 
    record = PredictionHistory.query.filter_by(
        id=record_id,
        user_id=session["user_id"]
    ).first()
    # show analysis  result single,file or youtube according to the record type 
    if not record:
        flash("Record not found", "error")
        return redirect(url_for("history"))

    if record.type == "single":
        return render_template(
           "view_single.html",
            record=record
        )
    elif record.type == "file":
        return render_template(
           "upload.html",
            history_record=record
        )

    elif record.type == "youtube":
        total = record.positive_count + record.negative_count + record.neutral_count 
        return render_template(
           "view_youtube_history.html",
            record=record,
             total=total,
             positive=record.positive_count,
             negative=record.negative_count,
             neutral=record.neutral_count,
             wordcloud_image=record.file_name
          )


#delete route for history
@app.route('/history/delete/<int:id>')
def delete_history(id):

    record = PredictionHistory.query.filter_by(
        id=id,
        user_id=session["user_id"]
    ).first()

    if record:
        db.session.delete(record)
        db.session.commit()

    return redirect('/history')



# This /predict route is specifically for Single Text Analysis (Dashboard prediction)
@app.route("/predict", methods=["POST"])
def predict():

    #Check user login
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    try:
        #Get JSON data safely
        data = request.get_json()

        # Check if JSON exists and contains text
        if not data or "text" not in data:
            return jsonify({"error": "No text provided"}), 400
         # Clean text (remove extra spaces)
        text = data.get("text").strip()
        #Check empty text
        if text == "":
            return jsonify({"error": "No text provided"}), 400
        #Handle short meaningless text
        if len(text.split()) < 2 or len(text) < 5:

            sentiment = "Neutral"
            confidence = 0
            color = "yellow"
            icon = "fa-meh"
            polarity = 0

            # Save prediction in database
            new_record = PredictionHistory(
                user_id=session["user_id"],
                date_time=datetime.now().strftime("%d %b %Y %I:%M %p"),
                type="single",
                input_text=text,
                sentiment=sentiment,
                confidence=confidence,
                polarity=polarity
            )

            db.session.add(new_record)
            db.session.commit()

            # Return result to frontend
            return jsonify({
                "sentiment": sentiment,
                "confidence": confidence,
                "color": color,
                "icon": icon,
                "polarity": polarity
            })


        # Normal prediction using main model
        sentiment, confidence, color, icon, polarity = predict_sentiment(text)
         #Save prediction in database
        new_record = PredictionHistory(
            user_id=session["user_id"],
            date_time=datetime.now().strftime("%d %b %Y %I:%M %p"),
            type="single",
            input_text=text,
            sentiment=sentiment,
            confidence=confidence,
            polarity=polarity
        )

        db.session.add(new_record)
        db.session.commit()
        #Send result to frontend
        return jsonify({
            "sentiment": sentiment,
            "confidence": confidence,
            "color": color,
            "icon": icon,
            "polarity": polarity
        })
     # Error handling
    except Exception as e:
        return jsonify({"error": str(e)}), 500




# logout route 
@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out successfully","success")
    return redirect(url_for("login"))


# upload route
@app.route('/upload')
def upload():
    return render_template('upload.html')


# instructions route
@app.route('/instructions')
def instructions():
    return render_template('instructions.html')


# YOUTUBE ANALYSIS CODE 
# this block of code featch only english comments from youtube video using LANGDETECT(detect the language of text automatically)library 
from langdetect import detect

def is_english(text):
    try:
        return detect(text) == "en"
    except:
        return False
# WE TAKE ONLY 100 COMMENTS
def fetch_youtube_comments(video_id, max_results=150):
    try:
        youtube = build('youtube', 'v3', developerKey=API_KEY)

        comments = []
        next_page_token = None

        while len(comments) < max_results:

            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=100,   # max allowed per request
                pageToken=next_page_token,
                textFormat="plainText"
            )

            response = request.execute()

            for item in response.get("items", []):
                comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                comments.append(comment)

                if len(comments) >= max_results:
                    break

            # Get next page token
            next_page_token = response.get("nextPageToken")

            # Stop if no more pages
            if not next_page_token:
                break

        return comments, None

    except HttpError as e:
        error_message = str(e)

        if "commentsDisabled" in error_message:
            return None, "Comments can't be fetched for this video."

        elif "videoNotFound" in error_message or "notFound" in error_message:
            return None, "This link does not exist. Try another one."

        else:
            return None, "Something went wrong. Please try another video."
# FETCH VIDEO TITLE
def fetch_video_title(video_id):
    try:
        youtube = build('youtube', 'v3', developerKey=API_KEY)

        request = youtube.videos().list(
            part="snippet",
            id=video_id
        )

        response = request.execute()

        items = response.get("items")

        if items:
            return items[0]["snippet"]["title"]

        return "Unknown Title"

    except Exception:
        return "Unknown Title"
# feach comments
def extract_video_id(url):
    import re
    pattern = r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})"
    match = re.search(pattern, url)
    if match:
        return match.group(1)
    return None
# CLEAN COMMENT TEXT YOUTUBE (MATCH TRAINING PREPROCESSING)
def clean_text(text):
    text = str(text)
    # lowercase (important because model trained on uncased)
    text = text.lower()
    # remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)
    # remove mentions
    text = re.sub(r"@\w+", "", text)
     # remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text
# SHORTEN COMMENT FOR DISPLAY ONLY (NOT FOR MODEL only display purpose)
def shorten_comment(comment, max_len=200):
    if len(comment) <= max_len:
        return comment
    return comment[:max_len] + "..."




# this is for youtube analysis page, accepts GET and POST requests, extracts video ID from YouTube link and renders analysis page with video ID for further processing
@app.route('/youtube_analysis', methods=['GET', 'POST'])
def youtube_analysis():
    if "user_id" not in session:
        flash("Please login first", "error")
        return redirect(url_for("login"))

    if request.method == 'POST':

        youtube_link = request.form['youtube_link']
        video_id = extract_video_id(youtube_link)

        if not video_id:
            flash("Invalid YouTube link", "error")
            return redirect(url_for("youtube_analysis"))

        all_comments, error = fetch_youtube_comments(video_id)
        video_title = fetch_video_title(video_id)

        if error:
            flash(error, "error")
            return redirect(url_for("youtube_analysis"))

        comments = []
        for comment in all_comments:
            if is_english(comment):
                cleaned_comment = clean_text(comment)
                if cleaned_comment != "":
                    comments.append(cleaned_comment)

        total_comments = len(comments)

        if total_comments == 0:
            flash("No English comments found for analysis.", "error")
            return redirect(url_for("youtube_analysis"))

               #  WORD CLOUD 
        text_data = " ".join(comments)

        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white'
        ).generate(text_data)

        filename = f"wordcloud_{session['user_id']}_{int(datetime.now().timestamp())}.png"
        wordcloud_path = os.path.join("static", filename)
        wordcloud.to_file(wordcloud_path)


        # SENTIMENT COUNTING
        positive_count = 0
        negative_count = 0
        neutral_count = 0

        # Lists to store top 3 comments per sentiment
        top_positive = []
        top_negative = []
        top_neutral = []

        # Loop through comments and collect sentiment + confidence
        for comment in comments:

            sentiment, confidence, color, icon, polarity = predict_sentiment(comment)

            short_comment = shorten_comment(comment)

            if sentiment == "Positive":
                positive_count += 1
                top_positive.append((short_comment, confidence))

            elif sentiment == "Negative":
                negative_count += 1
                top_negative.append((short_comment, confidence))

            else:
                neutral_count += 1
                top_neutral.append((short_comment, confidence))


        # Sort by confidence (highest first)
        top_positive = sorted(top_positive, key=lambda x: x[1], reverse=True)[:3]
        top_negative = sorted(top_negative, key=lambda x: x[1], reverse=True)[:3]
        top_neutral = sorted(top_neutral, key=lambda x: x[1], reverse=True)[:3]
        # Extract only comment text
        # Keep both comment and confidence
        top_positive = [(comment, round(conf,2)) for comment, conf in top_positive]
        top_negative = [(comment, round(conf,2)) for comment, conf in top_negative]
        top_neutral = [(comment, round(conf,2)) for comment, conf in top_neutral]
        #add in database
        new_record = PredictionHistory(
            user_id=session["user_id"],
            date_time=datetime.now().strftime("%d %b %Y %I:%M %p"),
            type="youtube",
            input_text=video_title,   # store title instead of id
            file_name=filename,
            positive_count=positive_count,
            negative_count=negative_count,
            neutral_count=neutral_count
        )

        db.session.add(new_record)
        db.session.commit()


        return render_template(
            'youtube_analysis.html',
            video_id=video_id,
            video_title=video_title,
            total=total_comments,
            positive=positive_count,
            negative=negative_count,
            neutral=neutral_count,
            wordcloud_image=filename,
            top_positive=top_positive,
            top_negative=top_negative,
            top_neutral=top_neutral
        )
    return render_template("youtube_analysis.html")



        
# PDF DOWNLOAD ROUTE FOR YOUTUBE ANALYSIS
#import libraries for PDF generation and chart creation
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib import colors

# PDF generation route for youtube analysis 
@app.route('/download_pdf', methods=['POST'])
def download_pdf():

    import json

    video_title = request.form.get("video_title", "Unknown Title")
    total = int(request.form.get("total", 0))
    positive = int(request.form.get("positive", 0))
    negative = int(request.form.get("negative", 0))
    neutral = int(request.form.get("neutral", 0))
    wordcloud_image = request.form.get("wordcloud_image")
    custom_name = request.form.get("custom_name")

    # GET TOP COMMENTS FROM FORM
    top_positive = json.loads(request.form.get("top_positive", "[]"))
    top_negative = json.loads(request.form.get("top_negative", "[]"))
    top_neutral = json.loads(request.form.get("top_neutral", "[]"))

    # Custom file name handling
    if custom_name and custom_name.strip() != "":
        safe_name = re.sub(r'[^a-zA-Z0-9_-]', '', custom_name)
        pdf_filename = f"{safe_name}.pdf"
    else:
        pdf_filename = f"youtube_report_{int(datetime.now().timestamp())}.pdf"

    pdf_path = os.path.join("static", pdf_filename)

    # CREATE PIE CHART IMAGE
    labels = ['Positive', 'Negative', 'Neutral']
    sizes = [positive, negative, neutral]
    colors_list = ['#198754','#dc3545','#ffc107']

    plt.figure(figsize=(4,4))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors_list)
    plt.title("Sentiment Distribution")

    pie_chart_path = os.path.join("static", "temp_pie_chart.png")
    plt.savefig(pie_chart_path)
    plt.close()

    # BUILD PDF
    doc = SimpleDocTemplate(pdf_path)
    elements = []
    styles = getSampleStyleSheet()

    # TITLE
    elements.append(Paragraph("<b>YouTube Sentiment Analysis Report</b>", styles['Title']))
    elements.append(Spacer(1, 0.5 * inch))

    # VIDEO INFO
    elements.append(Paragraph(f"<b>Video Title:</b> {video_title}", styles['Normal']))
    elements.append(Spacer(1, 0.3 * inch))

    # SUMMARY
    elements.append(Paragraph(f"Total Comments: {total}", styles['Normal']))
    elements.append(Paragraph(f"Positive: {positive}", styles['Normal']))
    elements.append(Paragraph(f"Negative: {negative}", styles['Normal']))
    elements.append(Paragraph(f"Neutral: {neutral}", styles['Normal']))
    elements.append(Spacer(1, 0.5 * inch))

    # PIE CHART
    elements.append(Paragraph("<b>Sentiment Distribution</b>", styles['Heading2']))
    elements.append(Spacer(1, 0.2 * inch))
    elements.append(Image(pie_chart_path, width=4*inch, height=4*inch))
    elements.append(Spacer(1, 0.5 * inch))

    # WORD CLOUD
    if wordcloud_image:
        elements.append(Paragraph("<b>Word Cloud</b>", styles['Heading2']))
        elements.append(Spacer(1, 0.2 * inch))
        wc_path = os.path.join("static", wordcloud_image)
        elements.append(Image(wc_path, width=5*inch, height=3*inch))
        elements.append(Spacer(1, 0.5 * inch))

    # TOP POSITIVE COMMENTS
    elements.append(Paragraph("<b>Top Positive Comments</b>", styles['Heading2']))
    elements.append(Spacer(1, 0.2 * inch))

    for comment, confidence in top_positive:
        elements.append(Paragraph(f"{comment}", styles['Normal']))
        elements.append(Paragraph(f"Confidence: {confidence}%", styles['Normal']))
        elements.append(Spacer(1, 0.2 * inch))

    # TOP NEGATIVE COMMENTS
    elements.append(Paragraph("<b>Top Negative Comments</b>", styles['Heading2']))
    elements.append(Spacer(1, 0.2 * inch))

    for comment, confidence in top_negative:
        elements.append(Paragraph(f"{comment}", styles['Normal']))
        elements.append(Paragraph(f"Confidence: {confidence}%", styles['Normal']))
        elements.append(Spacer(1, 0.2 * inch))

    # TOP NEUTRAL COMMENTS
    elements.append(Paragraph("<b>Top Neutral Comments</b>", styles['Heading2']))
    elements.append(Spacer(1, 0.2 * inch))

    for comment, confidence in top_neutral:
        elements.append(Paragraph(f"{comment}", styles['Normal']))
        elements.append(Paragraph(f"Confidence: {confidence}%", styles['Normal']))
        elements.append(Spacer(1, 0.2 * inch))

    # BUILD PDF
    doc.build(elements)

    return send_file(pdf_path, as_attachment=True)
  
# upload and analyze route, accepts POST request with CSV file, processes each row, and returns analysis results in JSON format
@app.route('/upload_analyze',methods=['POST'])
def upload_analyze():
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}),401
    file=request.files.get("file")
    if not file:
        return jsonify ({"error": "No file uploaded"}),400
    # handle file size limit (5MB)
    if file.content_length and file.content_length >5*1024*1024:
        return jsonify({"error": "File size exceeds 5MB limit"}),400
    filename=file.filename.lower()
    texts=[]
    # red txt file
    if filename.endswith('.txt'):
        content=file.read().decode('utf-8')
        lines=content.splitlines()
        for line in lines:
            if line.strip():
                texts.append(line.strip())
                
    # read csv file
    elif filename.endswith('.csv'):
        content=file.read().decode('utf-8')
        reader=csv.reader(io.StringIO(content))
        for row in reader:
            if row and row[0].strip():
                texts.append(row[0].strip())
    else:
        return jsonify({"error": "invalid file format"}),400
    #check if file is empty
    if len(texts)==0:
        return jsonify({"error":"file is empty"}),400
    # statistics variable assing
    total=len(texts)
    positive=0
    negative=0
    neutral=0

    conf_90_100=0
    conf_80_90=0
    conf_70_80=0
    # run model on each text and calculate statistics
    for text in texts:
        sentiment,confidence,color,icon,polarity=predict_sentiment(text)
        if sentiment=="Positive":
            positive+=1
        elif sentiment=="Negative":
            negative+=1
        else:
            neutral+=1
    # confidence distribution
        if confidence>=90:
            conf_90_100+=1
        elif confidence>=80:
            conf_80_90+=1
        else:
            conf_70_80+=1

    # WORD CLOUD GENERATION UPLOAD FILE ANALYSIS
    text_data = " ".join(texts)
    stopwords = set(STOPWORDS)

    wordcloud = WordCloud(
       width=800,
       height=400,
       background_color="white",
       stopwords=stopwords
).generate(text_data)
    filename = f"wordcloud_file_{session['user_id']}_{int(datetime.now().timestamp())}.png"

    wordcloud_path = os.path.join("static", filename)
    wordcloud.to_file(wordcloud_path)
    
    #  SAVE FILE ANALYSIS SUMMARY in database 
    new_record = PredictionHistory(
        user_id=session["user_id"],
        date_time=datetime.now().strftime("%d %b %Y %I:%M %p"),
        type="file",
        file_name=filename,
        positive_count=positive,
        negative_count=negative,
        neutral_count=neutral,
        conf_90_100=conf_90_100,
        conf_80_90=conf_80_90,
        conf_70_80=conf_70_80
    )

    db.session.add(new_record)
    db.session.commit()

 # result send to forntend
    return jsonify({
    "total": total,
    "positive": positive,
    "negative": negative,
    "neutral": neutral,
    "confidence": {
        "90_100": conf_90_100,
        "80_90": conf_80_90,
        "70_80": conf_70_80
    },
    "wordcloud": filename
})

#Running the app in debug mode
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)


