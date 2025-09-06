from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import customtkinter as CTk
from tkinter import END


def clearAll():
    # detecting the content from the entry box
    negativeField.delete(0,END)
    neutralField.delete(0,END)
    positiveField.delete(0, END)
    overallField.delete(0, END)

    # whole content of text area is deleted
    textArea.delete(1.0, END)

# functon to print sentiment of the sentence
def detect_sentiment():

    #get a whole input content from the box
    sentence = textArea.get("1.0", "end")
    # Create a SentimentIntensityAnalyzer() object
    sid_obj = SentimentIntensityAnalyzer()

    # polarity_scores method of SentimentIntensityAnalyzer
    # object gives a sentiment dictionary
    # which contains pos, neg, neu, and compound scores
    sentiment_dict = sid_obj.polarity_scores(sentence)

    string = str(sentiment_dict['neg']*100) + "% Negative"
    negativeField.insert(10, string)

    string = str(sentiment_dict['neu']*100) + "% Neutral" 
    neutralField.insert(10, string)

    string = str(sentiment_dict['pos']*100) + "% Posititve"
    positiveField.insert(10, string)

    # decide sentiment as positive, negative and neutral
    if sentiment_dict['compound'] >= 0.05:
        string = "Positive"

    elif sentiment_dict[''] <= -0.05:
        string = "Negative"

    else:
        string = "Neutral"

    overallField.insert(10, string)


def button_callback():
    print("Sentiment analysis started...")

if __name__ == "__main__":


    app = CTk.CTk()
    app.config(background = "light blue")

    app.title("Sentiment Detector")
    app.geometry("250x400")

    # create alabel: Enter your task
    enterText = CTk.CTkLabel(app, text = "Enter Your Sentence", fg_color="light blue")

    textArea = CTk.CTkTextbox(app, height = 15, width = 200, corner_radius=10, bg_color="light blue", pady=10)

    # create a submit button and place into the root window
    # when user presses the button, the command or function
    # affiliated to that button is executed

    check = CTk.CTkButton(app, text = "Check Sentiment", fg_color ="Red", bg_color="light blue", corner_radius=10, command = detect_sentiment)

    # Create a negative : label
    negative = CTk.CTkLabel(app, text="Sentence was rated as: ", fg_color="light blue", pady=10)

    # Create a neutral : label
    neutral = CTk.CTkLabel(app, text="Sentence was rated as: ", fg_color="light blue", pady=10)

    # Create a negative : label
    positive = CTk.CTkLabel(app, text="Sentence was rated as: ", fg_color="light blue", pady=10)

    # create a text entry Box
    negativeField = CTk.CTkEntry(app)

    # create a text entry Box
    neutralField = CTk.CTkEntry(app)

    # create a text entry Box
    positiveField = CTk.CTkEntry(app)

    # create a text entry Box
    overallField = CTk.CTkEntry(app)

    # clear button
    clear = CTk.CTkButton(app, text = "Clear", fg_color = "Red", bg_color ="light blue", corner_radius=10, command=clearAll)

    # Exit button

    Exit = CTk.CTkButton(app, text = "Exit", fg_color = "Red", bg_color ="light blue", corner_radius=10, command=exit)

    # grid method for placing the widgets at the respective positions

    enterText.grid(row = 0, column = 2)
    textArea.grid(row = 1, column = 2, padx = 10, sticky = "w")

    check.grid(row = 2, column = 2)
    negative.grid(row = 3, column = 2)
    positive.grid(row = 5, column = 2)
    neutral.grid(row = 7, column = 2)
    overallField.grid(row = 9, column = 2)

    negativeField.grid(row = 4, column = 2)
    neutralField.grid(row = 6, column = 2)
    positiveField.grid(row = 8, column = 2)
    overallField.grid(row = 10, column = 2)

    clear.grid(row=11, column=2)
    Exit.grid(row = 12, column=2)

    app.mainloop()