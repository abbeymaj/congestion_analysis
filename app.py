# Importing packages
import pandas as pd
from Flask import Flask, request, render_template, jsonify

# Creating the Flask app
app = Flask(__name__)