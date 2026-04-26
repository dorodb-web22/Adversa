# Hugging Face Space Deployment Guide

Follow these exact steps to push the `Adversa` environment to Hugging Face Spaces:

## Step 1: Create the Space on Hugging Face
1. Go to [Hugging Face Spaces](https://huggingface.co/spaces) (make sure you are logged in).
2. Click the **"Create new Space"** button in the top right.
3. Fill out the form:
   * **Space name**: `Adversa`
   * **License**: `MIT`
   * **Select the Space SDK**: Choose **Docker**.
   * **Docker Template**: Choose **Blank**.
   * **Space Hardware**: Free tier (CPU basic) is fine for the environment.
4. Click **"Create Space"**.

## Step 2: Push Your Code via Git
You have two options to get your code into the Space. **Option A (Terminal)** is highly recommended.

### Option A: Push via Terminal (Recommended)
Open your terminal and run these exact commands (replace `dorare22` with your actual Hugging Face username if it's different):

```bash
# Ensure you are in the project directory
cd /Users/rajesh/Desktop/Adversa

# Add the Hugging Face space as a remote repository
# Note: You will need your Hugging Face Access Token if prompted for a password.
# Get one at: https://huggingface.co/settings/tokens (needs 'write' role)
git remote add hf https://huggingface.co/spaces/dorare22/Adversa

# Push your code to Hugging Face
git push -u hf master:main
```
*(Note: We push your local `master` branch to the HF remote's `main` branch).*

### Option B: Manual Upload via Web UI (Fallback)
If the terminal push gives you Git credential headaches:
1. Go to your newly created Space on the web.
2. Click the **"Files"** tab.
3. Click **"Add file"** -> **"Upload files"**.
4. Drag and drop ALL the files from `/Users/rajesh/Desktop/Adversa` (including the `server` and `frontend` folders, `Dockerfile`, `run.sh`, etc.).
5. Add a commit message and click **"Commit changes"**.

## Step 3: Verify the Build
1. Once the code is pushed, go to the **"App"** tab on your Hugging Face Space.
2. You will see a "Building" status log. The `Dockerfile` will install the requirements and then execute `run.sh`.
3. Wait about 1-2 minutes.
4. When it says "Running", you should immediately see the beautiful navy/gold Gradio dashboard!
5. **Test the API**: The backend is also running. You can test it by going to `https://dorare22-adversa.hf.space/health` in your browser. (It should return a JSON response).
