
![title](https://github.com/smmin21/ACTemo-server/assets/79392773/4e7424d2-4f99-46c8-a0eb-982f530e9a8e)

<details>
<summary>Table of Contents</summary>

- [🔮 Features](#-features)
- [🛳 Running `ACTion` with Docker](#-running-action-with-docker)
  - [🚀 Make Your Own API Server](#-make-your-own-api-server)
    - [`1` API Request Example](#1-api-request-example)
    - [`2` API Response Example](#2-api-response-example)
- [🧩 Just Want to Test the Model](#-just-want-to-test-the-model)
  - [Execute Tests](#execute-tests)
- [🔗 Link to Other Parts](#-link-to-other-parts)

</details>


---

## 🔮 Features
**ACTemo**'s `ACTion` feature enables users to choose the emotional category they wish to practice and receive real-time feedback on the accuracy of their expression in the audio input. The feedback, provided in grades like 'A', 'B', and 'C', evaluates the expression's precision, ensuring users can refine their emotional portrayal skills instantly.


<br>

## 🛳 Running `ACTion` with Docker 
ACTemo provides a self-hosted version with `Dockerfile`. This allows you to deploy your own **speech-emotion-evaluation model** in just a few minutes, even without prior expertise.

### 🚀 Make Your Own API Server
We provide a Dockerfile for deploying the ACTemo's ACTion function on your own private device. Use the following command to start:
```bash
# 1. Build docker image
$ docker build -f SER_model/Dockerfile.prod -t actemo:prod .

# 2. Run!
$ docker run -it --rm --name actemo-prod -p 8000:8000 actemo:prod
```
#### **`1` API Request Example**
To make a request to the API server, use the following endpoint:
  
- **Method:** POST 
- **Host:** http://localhost:8000 
- **URL:** /predict 

In the request body, include the following keys and values: 
- Attach your file to the `"file"` key. 
- Specify one of the following emotions as the value for the `"emotion"` key:
```
'Angry', 'Distressed', 'Content', 'Happy', 'Excited', 'Joyous', 'Depressed', 'Sad', 'Bored', 'Calm', 'Relaxed', 'Sleepy', 'Afraid', 'Surprised'
```

> [!TIP]
> We support all audio formats compatible with ffmpeg, such as `m4a`, `mp3`, `caf`, and more.

#### **`2` API Response Example**
The response is returned in JSON format, with each audio assigned a grade, such as 'A', 'B', and 'C'.

<br>

---

### 🧩 Just Want to Test the Model
If you simply want to train or test the model without deploying the API server, follow these steps:
```bash
# 1. Build docker image
$ docker build -f SER_model/Dockerfile.dev -t actemo:dev .

# 2. Run!
$ docker run -it --rm --name actemo-dev -p 8000:8000 -v $(pwd)/SER_model:/root/code actemo:dev /bin/bash
```
> [!NOTE]
> After running the Docker container using the provided command, you'll have access to the `bash` shell within the container environment.


#### **Execute Tests**
Inside the Docker container's bash shell, execute the following command to run the test:

```bash
$ python test.py
```

You will be prompted with the following messages:

1. `"Enter the emotion: "`: Input one of the listed emotions.
	```
	'Angry', 'Distressed', 'Content', 'Happy', 'Excited', 'Joyous', 'Depressed', 'Sad', 'Bored', 'Calm', 'Relaxed', 'Sleepy', 'Afraid', 'Surprised'
	```

2. `"Enter the path of the audio file: "`: Input the path of the audio file containing the practiced emotional expression.

> [!TIP]
> This process will continue indefinitely. If you wish to stop, simply press `Ctrl + C`.

<br>

## 🔗 Link to Other Parts

If you want to explore other parts of the project, feel free to navigate to:

- [Overall Project Repository](https://github.com/smmin21/ACTemo-Google-Solution-Challenge-2024): For the complete project repository, including all parts, visit the GitHub repository.
- [Frontend Development](https://github.com/e6d1fe/ACTemo-flutter.git): Explore the frontend development aspects of the project, including UI design and user interaction, implemented using Flutter.

<br>
