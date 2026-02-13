# import necessary libraries
from flask import Flask, render_template, request, redirect, url_for, flash,session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import re
import os
#creating flask app and secret key
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "sentimodel_key_for_session")

# Database  configaration
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///users.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
#intializing database
db = SQLAlchemy(app)

# Database Model(user table)
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)

# Create DB automatically,Creates users.db,Creates "user" table if not exists,
with app.app_context():
    db.create_all()

# Home Routes
@app.route("/")
def home():
    return render_template("home.html")
# Registration route GET=>SHOWS FORM ,POST=>process form data,getting form data 
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

# login route 
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

#dashboard route
@app.route("/dashboard")
def dashboard():
    if "user_id" not in session:
        flash("Please login first","error")
        return redirect(url_for("login"))
    return render_template("dashboard.html")

# logout route 
@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out successfully","success")
    return redirect(url_for("login"))
#Running the app in debug mode
if __name__ == "__main__":
    app.run(debug=True)

