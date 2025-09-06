from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import customtkinter as CTk

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

    string = str(sentiment_dict['neu']*100) + "% Neutral" 
    neutralField.insert(10, string)

    string = str(sentiment_dict['pos']*100) + "%Posititve"
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
    app.config(background = "light green")

    app.title("Sentiment Detector")
    app.geometry("400x150")

    # create alabel: Enter your task
    enterText = CTk.CTkLabel(app, text = "Enter Your Sentence", fg_color="light green")

    #textArea = CTk.CTkText(app, heihgt = 5, width = 25, font ="lucida 13")


    button = CTk.CTkButton(app, text="my button", command=button_callback)
    button.pack(padx=20, pady=20)


    app.mainloop()