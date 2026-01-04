This guide is designed for developers of an open-source AI agent who need to provide clear, step-by-step instructions for their users to set up the necessary Google credentials. This approach minimizes user friction by having them create a single **Desktop app** client and run a seamless authentication script.

## 🗃️ AI Agent Gmail Toolkit: User Setup Guide

### Part 1: Setting up the Google Cloud Console (One-Time Developer Setup)

Each user must perform this setup to create their own secure credentials. You must guide them to create a **Desktop app** client, which is crucial for the simple, no-server authentication flow.

#### Step 1: Create a Google Cloud Project

A project is a container for your app's settings and APIs.

1. Go to the **Google Cloud Console** and sign in.
2. Click the **"Select a project"** dropdown at the top.
3. Click **"New Project"** and give it a name (e.g., "My AI Agent Gmail Credentials").
4. Select the newly created project as the active project.

#### Step 2: Enable the Gmail API

You must explicitly enable the Gmail API for the project to function.

1. In the Console, go to **APIs & Services** > **Library**.
2. Search for **"Gmail API"**.
3. Click the result and click the **"Enable"** button.

#### Step 3: Configure the OAuth Consent Screen

This screen informs users what data your agent is requesting access to.

1. In the Console, navigate to **APIs & Services** > **OAuth Consent Screen**.
2. Select **"External"** as the User Type (since it's for external Google accounts).
3. Complete the required fields:
* **App name:** The name of your AI agent (e.g., "My AI Email Agent").
* **User support email:** A contact email.
* **Developer contact information:** Your email.
4. Click **"Save and Continue."**

#### Step 4: Define the Scopes (Permissions)

This step tells Google the maximum permissions your app can ever request.

1. On the **Data access** page, click **"Add or Remove Scopes."**
2. Manually add the following two scopes by searching or pasting them:
* `.../auth/gmail.modify`
* `.../auth/gmail.compose`
3. Click **"Update."** These scopes enable all your planned operations:
* **`gmail.modify`**: Read messages/threads, search, modify labels, and trash messages.
* **`gmail.compose`**: Create, retrieve, update, and send drafts/messages.
4. Click **"Save and Continue"** to review the summary and return to the dashboard.

#### Step 5: Create and Download the Client Secret JSON

This is the final step, generating the credential file that the AI agent uses to identify itself to Google.

1. In the Console, go to **APIs & Services** > **Credentials**.
2. Click **"+ Create Credentials"** and select **"OAuth client ID"**.
3. In the **Application type** dropdown, select **"Desktop app"**.
*Technical note:* This type is vital because it enables the simple Loopback IP flow, meaning the user won't need to configure a redirect URI or run a local server.
4. Give it a simple name (e.g., "Agent Desktop Client") and click **"Create."**
5. A dialog box will appear. Click **"Download JSON"** and save the file.

**SAVE THIS FILE:** The user must rename this file to **`client_secret.json`** and place it in the root directory of your AI agent's project. This is the **only file** they need from the Console setup.

#### Step 6: Whitelist Your Account (Test Users)

Because your app is currently in "Testing" mode, Google will block all login attempts unless the email address is explicitly authorized.

1. In the Google Cloud Console, navigate to **APIs & Services** > **OAuth Consent Screen**.
(Note: In the new layout, this is under Google Auth Platform > Audience).
2. Scroll down to the **"Test users"** section and click **"+ ADD USERS"**.
3. Enter your Gmail address (the one you will use to send emails).
4. Click **"Save"**.

*Important:* When you run the script, you will see a "Google hasn't verified this app" warning. Click **"Advanced"** and then **"Go to [App Name] (unsafe)"** to proceed.

---

### Part 2: Agent Setup and First-Run Authorization

This process is handled entirely by your AI agent's Python code (using the Loopback IP flow), requiring only one manual action from the user.

#### Step 1: Place the Credentials File

Place the **`client_secret.json`** file (downloaded in Part 1, Step 5) into the main directory of the AI Agent.

#### Step 2: Run the Authorization Script

The user runs the provided setup script (e.g., `python auth.py`).

#### Step 3: The Seamless Handshake

1. **Browser Opens:** The Python script will automatically open the user's default web browser to the Google Consent Screen URL.
2. **User Grants Permission:** The user signs into their Google Account and clicks **"Allow"** to grant your agent the requested **`gmail.modify`** and **`gmail.compose`** scopes.
3. **Automatic Capture:** The Python code automatically spins up a temporary server on a local port, captures the Authorization Code from the redirect, and silently exchanges it for the **Refresh Token**.
4. **Token Saved:** The script automatically saves the Refresh Token to a file named **`token.json`** on the user's system.

#### Step 4: The Agent is Ready

The **`token.json`** file is the permanent key. For all future operations, your agent will use this file to get new Access Tokens without ever requiring the user to open a browser again. The setup is complete.

---

### Part 3: Moving to Production (Optional: Prevent Token Expiry)

By default, apps in "Testing" mode have refresh tokens that expire every 7 days, forcing you to re-authenticate weekly. To make your token.json permanent, you must "Publish" the app.

#### How to Publish Your App

1. Go to the **OAuth Consent Screen** (or **Google Auth Platform**) page.
2. Under the **"Publishing status"** section, click the **"PUBLISH APP"** button.
3. A dialog will appear asking for confirmation. Click **"Confirm"**.
Note: Since you are using this as a personal Desktop App for yourself, you do not need to submit for actual verification or provide a privacy policy. Status: "In production" just means your personal tokens won't expire.
4. Once the status changes to **"In Production"**, delete your old **`token.json`** file and run your authentication script one last time. This new token will remain valid indefinitely unless you manually revoke it.
