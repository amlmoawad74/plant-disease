{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyMWhSSLOCaSrg1AiM3Lpz26",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/amlmoawad74/plant-disease/blob/main/app.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 1,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "F1geE2Yr6zlV",
        "outputId": "2490c9a6-8b75-44af-9c9d-d447c7d2028f"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Requirement already satisfied: streamlit in /usr/local/lib/python3.10/dist-packages (1.30.0)\n",
            "Requirement already satisfied: altair<6,>=4.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (4.2.2)\n",
            "Requirement already satisfied: blinker<2,>=1.0.0 in /usr/lib/python3/dist-packages (from streamlit) (1.4)\n",
            "Requirement already satisfied: cachetools<6,>=4.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (5.3.2)\n",
            "Requirement already satisfied: click<9,>=7.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (8.1.7)\n",
            "Requirement already satisfied: importlib-metadata<8,>=1.4 in /usr/local/lib/python3.10/dist-packages (from streamlit) (7.0.1)\n",
            "Requirement already satisfied: numpy<2,>=1.19.3 in /usr/local/lib/python3.10/dist-packages (from streamlit) (1.23.5)\n",
            "Requirement already satisfied: packaging<24,>=16.8 in /usr/local/lib/python3.10/dist-packages (from streamlit) (23.2)\n",
            "Requirement already satisfied: pandas<3,>=1.3.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (1.5.3)\n",
            "Requirement already satisfied: pillow<11,>=7.1.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (9.4.0)\n",
            "Requirement already satisfied: protobuf<5,>=3.20 in /usr/local/lib/python3.10/dist-packages (from streamlit) (3.20.3)\n",
            "Requirement already satisfied: pyarrow>=6.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (10.0.1)\n",
            "Requirement already satisfied: python-dateutil<3,>=2.7.3 in /usr/local/lib/python3.10/dist-packages (from streamlit) (2.8.2)\n",
            "Requirement already satisfied: requests<3,>=2.27 in /usr/local/lib/python3.10/dist-packages (from streamlit) (2.31.0)\n",
            "Requirement already satisfied: rich<14,>=10.14.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (13.7.0)\n",
            "Requirement already satisfied: tenacity<9,>=8.1.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (8.2.3)\n",
            "Requirement already satisfied: toml<2,>=0.10.1 in /usr/local/lib/python3.10/dist-packages (from streamlit) (0.10.2)\n",
            "Requirement already satisfied: typing-extensions<5,>=4.3.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (4.5.0)\n",
            "Requirement already satisfied: tzlocal<6,>=1.1 in /usr/local/lib/python3.10/dist-packages (from streamlit) (5.2)\n",
            "Requirement already satisfied: validators<1,>=0.2 in /usr/local/lib/python3.10/dist-packages (from streamlit) (0.22.0)\n",
            "Requirement already satisfied: gitpython!=3.1.19,<4,>=3.0.7 in /usr/local/lib/python3.10/dist-packages (from streamlit) (3.1.41)\n",
            "Requirement already satisfied: pydeck<1,>=0.8.0b4 in /usr/local/lib/python3.10/dist-packages (from streamlit) (0.8.1b0)\n",
            "Requirement already satisfied: tornado<7,>=6.0.3 in /usr/local/lib/python3.10/dist-packages (from streamlit) (6.3.2)\n",
            "Requirement already satisfied: watchdog>=2.1.5 in /usr/local/lib/python3.10/dist-packages (from streamlit) (3.0.0)\n",
            "Requirement already satisfied: entrypoints in /usr/local/lib/python3.10/dist-packages (from altair<6,>=4.0->streamlit) (0.4)\n",
            "Requirement already satisfied: jinja2 in /usr/local/lib/python3.10/dist-packages (from altair<6,>=4.0->streamlit) (3.1.3)\n",
            "Requirement already satisfied: jsonschema>=3.0 in /usr/local/lib/python3.10/dist-packages (from altair<6,>=4.0->streamlit) (4.19.2)\n",
            "Requirement already satisfied: toolz in /usr/local/lib/python3.10/dist-packages (from altair<6,>=4.0->streamlit) (0.12.0)\n",
            "Requirement already satisfied: gitdb<5,>=4.0.1 in /usr/local/lib/python3.10/dist-packages (from gitpython!=3.1.19,<4,>=3.0.7->streamlit) (4.0.11)\n",
            "Requirement already satisfied: zipp>=0.5 in /usr/local/lib/python3.10/dist-packages (from importlib-metadata<8,>=1.4->streamlit) (3.17.0)\n",
            "Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.10/dist-packages (from pandas<3,>=1.3.0->streamlit) (2023.3.post1)\n",
            "Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.10/dist-packages (from python-dateutil<3,>=2.7.3->streamlit) (1.16.0)\n",
            "Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.27->streamlit) (3.3.2)\n",
            "Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.27->streamlit) (3.6)\n",
            "Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.27->streamlit) (2.0.7)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.27->streamlit) (2023.11.17)\n",
            "Requirement already satisfied: markdown-it-py>=2.2.0 in /usr/local/lib/python3.10/dist-packages (from rich<14,>=10.14.0->streamlit) (3.0.0)\n",
            "Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /usr/local/lib/python3.10/dist-packages (from rich<14,>=10.14.0->streamlit) (2.16.1)\n",
            "Requirement already satisfied: smmap<6,>=3.0.1 in /usr/local/lib/python3.10/dist-packages (from gitdb<5,>=4.0.1->gitpython!=3.1.19,<4,>=3.0.7->streamlit) (5.0.1)\n",
            "Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.10/dist-packages (from jinja2->altair<6,>=4.0->streamlit) (2.1.4)\n",
            "Requirement already satisfied: attrs>=22.2.0 in /usr/local/lib/python3.10/dist-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (23.2.0)\n",
            "Requirement already satisfied: jsonschema-specifications>=2023.03.6 in /usr/local/lib/python3.10/dist-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (2023.12.1)\n",
            "Requirement already satisfied: referencing>=0.28.4 in /usr/local/lib/python3.10/dist-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (0.32.1)\n",
            "Requirement already satisfied: rpds-py>=0.7.1 in /usr/local/lib/python3.10/dist-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (0.17.1)\n",
            "Requirement already satisfied: mdurl~=0.1 in /usr/local/lib/python3.10/dist-packages (from markdown-it-py>=2.2.0->rich<14,>=10.14.0->streamlit) (0.1.2)\n",
            "Requirement already satisfied: torch in /usr/local/lib/python3.10/dist-packages (2.1.0+cu121)\n",
            "Requirement already satisfied: torchvision in /usr/local/lib/python3.10/dist-packages (0.16.0+cu121)\n",
            "Requirement already satisfied: filelock in /usr/local/lib/python3.10/dist-packages (from torch) (3.13.1)\n",
            "Requirement already satisfied: typing-extensions in /usr/local/lib/python3.10/dist-packages (from torch) (4.5.0)\n",
            "Requirement already satisfied: sympy in /usr/local/lib/python3.10/dist-packages (from torch) (1.12)\n",
            "Requirement already satisfied: networkx in /usr/local/lib/python3.10/dist-packages (from torch) (3.2.1)\n",
            "Requirement already satisfied: jinja2 in /usr/local/lib/python3.10/dist-packages (from torch) (3.1.3)\n",
            "Requirement already satisfied: fsspec in /usr/local/lib/python3.10/dist-packages (from torch) (2023.6.0)\n",
            "Requirement already satisfied: triton==2.1.0 in /usr/local/lib/python3.10/dist-packages (from torch) (2.1.0)\n",
            "Requirement already satisfied: numpy in /usr/local/lib/python3.10/dist-packages (from torchvision) (1.23.5)\n",
            "Requirement already satisfied: requests in /usr/local/lib/python3.10/dist-packages (from torchvision) (2.31.0)\n",
            "Requirement already satisfied: pillow!=8.3.*,>=5.3.0 in /usr/local/lib/python3.10/dist-packages (from torchvision) (9.4.0)\n",
            "Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.10/dist-packages (from jinja2->torch) (2.1.4)\n",
            "Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests->torchvision) (3.3.2)\n",
            "Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests->torchvision) (3.6)\n",
            "Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests->torchvision) (2.0.7)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests->torchvision) (2023.11.17)\n",
            "Requirement already satisfied: mpmath>=0.19 in /usr/local/lib/python3.10/dist-packages (from sympy->torch) (1.3.0)\n"
          ]
        }
      ],
      "source": [
        "!pip install streamlit\n",
        "!pip install torch torchvision"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import streamlit as st\n",
        "from PIL import Image\n",
        "import torch\n",
        "import torch.nn as nn\n",
        "import torchvision.transforms as transforms"
      ],
      "metadata": {
        "id": "8EqnU2Ra62XE"
      },
      "execution_count": 2,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "class ImageClassificationBase(nn.Module):\n",
        "\n",
        "    def training_step(self, batch):\n",
        "        images, labels = batch\n",
        "        out = self(images)                  # Generate predictions\n",
        "        loss = F.cross_entropy(out, labels) # Calculate loss\n",
        "        return loss\n",
        "\n",
        "    def validation_step(self, batch):\n",
        "        images, labels = batch\n",
        "        out = self(images)                   # Generate prediction\n",
        "        loss = F.cross_entropy(out, labels)  # Calculate loss\n",
        "        acc = accuracy(out, labels)          # Calculate accuracy\n",
        "        return {\"val_loss\": loss.detach(), \"val_accuracy\": acc}\n",
        "\n",
        "    def validation_epoch_end(self, outputs):\n",
        "        batch_losses = [x[\"val_loss\"] for x in outputs]\n",
        "        batch_accuracy = [x[\"val_accuracy\"] for x in outputs]\n",
        "        epoch_loss = torch.stack(batch_losses).mean()       # Combine loss\n",
        "        epoch_accuracy = torch.stack(batch_accuracy).mean()\n",
        "        return {\"val_loss\": epoch_loss, \"val_accuracy\": epoch_accuracy} # Combine accuracies\n",
        "\n",
        "    def epoch_end(self, epoch, result):\n",
        "        print(\"Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}\".format(\n",
        "            epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_accuracy']))"
      ],
      "metadata": {
        "id": "YydS3GlvF4nS"
      },
      "execution_count": 3,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "import torch\n",
        "from torchvision import transforms\n",
        "from PIL import Image\n",
        "import streamlit as st\n",
        "\n",
        "# Define the class names for plant diseases\n",
        "classes = ['Tomato___Late_blight', 'Tomato___healthy', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Potato___healthy', 'Corn_(maize)___Northern_Leaf_Blight', 'Tomato___Early_blight', 'Tomato___Septoria_leaf_spot', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Strawberry___Leaf_scorch', 'Peach___healthy', 'Apple___Apple_scab', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Bacterial_spot', 'Apple___Black_rot', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Peach___Bacterial_spot', 'Apple___Cedar_apple_rust', 'Tomato___Target_Spot', 'Pepper,_bell___healthy', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Potato___Late_blight', 'Tomato___Tomato_mosaic_virus', 'Strawberry___healthy', 'Apple___healthy', 'Grape___Black_rot', 'Potato___Early_blight', 'Cherry_(including_sour)___healthy', 'Corn_(maize)___Common_rust_', 'Grape___Esca_(Black_Measles)', 'Raspberry___healthy', 'Tomato___Leaf_Mold', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Pepper,_bell___Bacterial_spot', 'Corn_(maize)___healthy']\n",
        "\n",
        "# Define the transformation for the input image\n",
        "transform = transforms.Compose([\n",
        "    transforms.Resize((224, 224)),\n",
        "    transforms.ToTensor(),\n",
        "])\n",
        "\n",
        "# Load the saved model as an OrderedDict\n",
        "checkpoint = torch.load('plant-disease-model (2).pth', map_location=torch.device('cpu'))\n",
        "\n",
        "# Check if the checkpoint is an OrderedDict\n",
        "if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:\n",
        "    # Extract the state dictionary from the checkpoint\n",
        "    state_dict = checkpoint['state_dict']\n",
        "else:\n",
        "    # The checkpoint is likely the state dictionary itself\n",
        "    state_dict = checkpoint\n",
        "\n",
        "# Create an instance of your model\n",
        "# (Replace 'YourModelClass' with the actual class name of your model)\n",
        "model = ImageClassificationBase()\n",
        "\n",
        "# Load the state dictionary into the model\n",
        "# Assuming 'model' is an instance of ImageClassificationBase\n",
        "filtered_state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict()}\n",
        "model.load_state_dict(filtered_state_dict)\n",
        "\n",
        "\n",
        "# Set the model to evaluation mode\n",
        "model.eval()\n",
        "\n",
        "def classify_image(image):\n",
        "    # Apply the transformation to the image\n",
        "    image_tensor = transform(image).unsqueeze(0)\n",
        "\n",
        "    # Make a prediction\n",
        "    with torch.no_grad():\n",
        "        outputs = model(image_tensor)\n",
        "        _, predicted = torch.max(outputs, 1)\n",
        "        class_index = predicted.item()\n",
        "\n",
        "    return class_index\n",
        "\n",
        "st.title(\"Plant Disease Classification\")\n",
        "\n",
        "image_file = st.file_uploader(\"Upload Image\", type=[\"png\", \"jpg\", \"jpeg\"])\n",
        "\n",
        "if image_file is not None:\n",
        "    image = Image.open(image_file)\n",
        "    st.image(image, width=250)\n",
        "    class_index = classify_image(image)\n",
        "    class_name = classes[class_index]\n",
        "    st.header(class_name)\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Yb8jx7o77AiC",
        "outputId": "ccdc2c8d-55c4-4d24-9597-7efad6b71592"
      },
      "execution_count": 4,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "2024-01-26 21:27:23.145 \n",
            "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
            "  command:\n",
            "\n",
            "    streamlit run /usr/local/lib/python3.10/dist-packages/colab_kernel_launcher.py [ARGUMENTS]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!streamlit run /usr/local/lib/python3.10/dist-packages/colab_kernel_launcher.py"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "5KYG8t0NiwB2",
        "outputId": "12480d5b-2559-4623-8d3d-567faa3f0725"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "Collecting usage statistics. To deactivate, set browser.gatherUsageStats to False.\n",
            "\u001b[0m\n",
            "\u001b[0m\n",
            "\u001b[34m\u001b[1m  You can now view your Streamlit app in your browser.\u001b[0m\n",
            "\u001b[0m\n",
            "\u001b[34m  Network URL: \u001b[0m\u001b[1mhttp://172.28.0.12:8501\u001b[0m\n",
            "\u001b[34m  External URL: \u001b[0m\u001b[1mhttp://35.221.2.163:8501\u001b[0m\n",
            "\u001b[0m\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "INIthcuYjTYr"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}