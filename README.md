# ezuthu
Tamil Character Classifier

### What is this project?

This project attempts to use a simple Pytorch vision learner applied on top of the [HPL Tamil Dataset](http://shiftleft.com/mirrors/www.hpl.hp.com/india/research/penhw-resources/TamilChar.pdf) to recognize handwritten tamil characters with reasonable accuracy. Refer to the notebook above to know more about the model and how it was trained.

A snapshot of what can be done with the app:

![](https://github.com/rcdexta/ezuthu/raw/master/assets/example.png)

The application takes user input which is handwritten on the canvas and compares it against its training data to make 3 best guesses about the character. 

### Learning Model

<pending>

### Server

The server is a simple python app on top of [Starlette](https://www.starlette.io/) that accepts a png image as multi-part form data and does inference against the model that we built using our training. Refer to `export.pkl` which is actually our model data that was output of our training. This is used as baseline by the server and it only uses CPU to do the inference. 


### Application

The application is a fork of [create-react-app](https://reactjs.org/docs/create-a-new-react-app.html) that contains a canvas that allows the user to draw the alphabet using a simple brush. On submit action, the app converts the canvas content to a PNG image and sends to the server. The server does the inference and based on predictions, sends the top 3 matching results for the given pixels. 

### Deployment

The app should be built separately and linked to the root `build` folder. The server will serve the app as a static asset. Both the app and server can be deployed in any containerized environment.
