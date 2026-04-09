from flask import Flask, Response
import subprocess
import os

app = Flask(__name__)

@app.route('/')
def home():
    """
    When someone visits the Render URL, this runs our main pipeline
    and returns exactly what the terminal would show, formatted as plain text.
    """
    try:
        # We use subprocess to run the exact same script you run locally
        result = subprocess.run(
            ["python", "main.py"], 
            capture_output=True, 
            text=True, 
            encoding='utf-8',
            check=True
        )
        output = result.stdout
    except subprocess.CalledProcessError as e:
        output = "An error occurred while running the pipeline:\n\n"
        output += e.stdout + "\n" + e.stderr
    except Exception as e:
        output = f"System Error: {str(e)}"

    # Return as text/plain so the browser preserves the nice spacing and formatting
    return Response(output, mimetype='text/plain')

if __name__ == '__main__':
    # Render assigns a dynamic port via the PORT environment variable
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
