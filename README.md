# facial-expression-recognition
A tutorial on how to deploy a simple facial expression recognition model on local machine, Heroku and AWS.

# 1. Setting up
Setting up a machine you sit in front of or SSH into easily.

## i. Check out the repo
```
git clone git@github.com:Chien10/facial-expression-recognition.git
cd facial-expression-recognition
```

## ii. Setting up the Python environment
- I use `conda` to manage Python version and use `Makefile` to make set-up straightforward. You can read more about `Makefile` [here](https://madewithml.com/courses/mlops/makefile/).  
- Conda is an open-source package management system and environment management system running on Linux, macOS and Windows. To install `conda`, you follow this [instruction](https://conda.io/projects/conda/en/latest/user-guide/install/) from the official website. Close and re-open your terminal after installing and check if `conda` command is valid.  
- The `Makefile` gives you the ability to run command defined within with with `make <command name>`. I encourage you to take a look at the [file](https://github.com/Chien10/facial-expression-recognition.git/Makefile). Run the following command to create an environment named `fer` (you can guess what it's short for!):
```
make conda-update
```
If you edit the [environment.yml](https://github.com/Chien10/facial-expression-recognition.git/environment.yml), just run the above command again to get the latest changes.  
- Next, activate the environment:
```
conda activate fer
```
Every time you work in this directory, remember to start your session with the previous command.  
- Eventually, add `export PYTHONPATH=.:$PYTHONPATH` to your `~/.bashrc` so that you can import packages defined.

# 2. Local deployment
- It's easy to run the application on your local machine.
- After finishing the [Setting up section](#1.-Sectting-up), set `FLASK_APP=app.py` with `set FLASK_APP=app.py` on your shell.
- Then, set `FLASK_ENV=development` with the same command.
- Now you can launch your app with `flask run` and enjoy the app at `http://127.0.0.1:5000/` or `http://localhost:5000/`.

# 3. Heroku deployment
- Now we'll move to a next level: deploying your app to a service from your local machine.
- Follow the subsequent steps to deploy your app to `Heroku`:
1. Make sure your project is tracked by [Git](https://git-scm.com/).
2. Install [Heroku CLI](https://devcenter.heroku.com/articles/heroku-cli).
3. Login to `Heroku` via this command: `heroku login`.
4. Create a new empty application on `Heroku` with: `heroku create`.
5. You can use `git remote -v` after the fourth step to confirm the remote named `heroku` has been set for your app.
6. To deploy the app, push the repo to the remote branch we just made: `git push heroku master`.
7. Check if the [dynos](https://www.heroku.com/dynos) is running with `heroku ps`.
8. If all the previous eight steps were finished successfully, you can enjoy your app now at the URL provided by `Heroku`. (If you have a problem finding the URL, look at the line saying something like this: *https://vast-harbor-73788.herokuapp.com/ deployed to Heroku* on the shell).
9. (Optional) To prevent traffics coming to the app: `heroku maintenance:on`.
10. (Optional) To completely stop the app: `heroku ps:scale web=0`. Make sure to turn off other process types defined in `Procfile`. If you just want to turn the app off for error fixing, remember to turn it on later with: `heroku ps:scale web=0`.

# 4. Docker
- You can skip the setting-up part with `conda` by using `Docker` which is another way ensures that the Python version is correct, install dependencies, check out the whole repo, cuda version, etc. Virtual environment is not enough when it comes to gpu version and even though this tutorial does not require cuda, it's convenient to use `Docker`.
- Install docker with this [instruction](https://docs.docker.com/get-docker/) from the Docker's website.
- Stay in the current directory, run: `docker build -t fer:1.0 -f api_server/Dockerfile .`.
- Inspect all the images and their attributes with: `docker images`.
- You can run the server with: `docker run -p 5000:5000 --name fer fer:1.0`.
- You can inspect all running and stopped containers: `docker ps` and `docker ps -a`.
- Your app is running on port 5000, make sure the service is active with: `sudo lsof -i -P -n | grep LISTEN`.
- When you've done with the app, stop the running container: `docker stop <CONTAINER_ID>`. If you want to remove it: `docker rm <CONTAINER_ID>`.
- You can remove a Docker image with: `docker image rm <IMAGE_NAME>`.

# 5. AWS deployment
## 5.1. Server

## 5.2. Serverless
