# Module for interacting with the Automatic1111 Stable Diffusion Web UI API.
import requests
import base64
import json
from typing import Dict, Any, Optional, Tuple

class A1111Client:
    """
    A client to interact with the Automatic1111 Stable Diffusion Web UI API.
    """
    def __init__(self, api_url: str):
        """
        Initializes the A1111Client.

        Args:
            api_url: The base URL of the A1111 API (e.g., "http://127.0.0.1:7860").
                     Should not end with a slash.
        """
        if api_url.endswith('/'):
            api_url = api_url[:-1] # Remove trailing slash if present
        self.base_url = api_url
        self.txt2img_url = f"{self.base_url}/sdapi/v1/txt2img"
        self._check_api_availability()

    def _check_api_availability(self):
        """Checks if the A1111 API is reachable."""
        try:
            # Use a simple endpoint like /sdapi/v1/progress to check connection
            response = requests.get(f"{self.base_url}/sdapi/v1/progress", timeout=5)
            response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
            print(f"Successfully connected to A1111 API at {self.base_url}")
        except requests.exceptions.ConnectionError:
            raise ConnectionError(f"Could not connect to A1111 API at {self.base_url}. Is it running?")
        except requests.exceptions.Timeout:
            raise TimeoutError(f"Connection to A1111 API at {self.base_url} timed out.")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Error checking A1111 API status: {e}")

    def generate_image(self, payload: Dict[str, Any]) -> Tuple[Optional[bytes], Optional[Dict[str, Any]]]:
        """
        Sends a request to the txt2img endpoint to generate an image.

        Args:
            payload: A dictionary containing the parameters for the txt2img API call.
                     See A1111 API documentation for details.
                     Example keys: 'prompt', 'negative_prompt', 'seed', 'steps', 'cfg_scale', etc.

        Returns:
            A tuple containing:
            - The generated image as bytes (if successful, otherwise None).
            - The 'info' dictionary from the API response containing generation metadata
              (if successful, otherwise None). Returns None if the API response format is unexpected.

        Raises:
            requests.exceptions.RequestException: If the API request fails.
            ValueError: If the API response is not as expected.
        """
        print(f"Sending generation request to {self.txt2img_url}...")
        # print(f"Payload: {json.dumps(payload, indent=2)}") # Uncomment for debugging

        try:
            response = requests.post(url=self.txt2img_url, json=payload, timeout=300) # 5 min timeout
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        except requests.exceptions.Timeout:
            print("Error: Request to A1111 API timed out.")
            return None, None
        except requests.exceptions.RequestException as e:
            print(f"Error during A1111 API request: {e}")
            # Attempt to get more details from the response if available
            try:
                error_details = response.json()
                print(f"API Error Details: {json.dumps(error_details, indent=2)}")
            except (AttributeError, json.JSONDecodeError):
                print("Could not decode error details from API response.")
            return None, None

        try:
            r = response.json()
            if 'images' not in r or not isinstance(r['images'], list) or len(r['images']) == 0:
                print("Error: 'images' field missing or invalid in API response.")
                print(f"Full response: {json.dumps(r, indent=2)}")
                return None, None

            # Decode the first image (assuming batch size 1 for now)
            image_data = base64.b64decode(r['images'][0])

            # Extract generation info (metadata)
            info_str = r.get('info', '{}') # Get info string, default to empty JSON string
            try:
                info_dict = json.loads(info_str)
            except json.JSONDecodeError:
                print("Warning: Could not parse 'info' JSON string from API response.")
                print(f"Info string received: {info_str}")
                info_dict = {} # Return empty dict if parsing fails

            print("Image generated successfully.")
            return image_data, info_dict

        except (json.JSONDecodeError, KeyError, IndexError, base64.binascii.Error) as e:
            print(f"Error processing A1111 API response: {e}")
            print(f"Full response: {response.text}") # Print raw response text for debugging
            return None, None
        except Exception as e:
            print(f"An unexpected error occurred while processing the response: {e}")
            return None, None


if __name__ == '__main__':
    # Example usage (requires a running A1111 instance with API enabled)
    print("Testing A1111Client...")
    # Make sure A1111 is running at this address and API is enabled
    # (e.g., run with --api flag)
    test_api_url = "http://127.0.0.1:7860"

    try:
        client = A1111Client(test_api_url)

        test_payload = {
            "prompt": "a cute kitten",
            "steps": 5, # Low steps for quick testing
            "seed": 123,
            "width": 256, # Small size for quick testing
            "height": 256
        }

        print(f"\nSending test payload: {json.dumps(test_payload, indent=2)}")
        image_bytes, info = client.generate_image(test_payload)

        if image_bytes:
            print("\nTest image generated successfully!")
            # Save the test image
            test_output_path = "test_a1111_output.png"
            with open(test_output_path, "wb") as f:
                f.write(image_bytes)
            print(f"Test image saved to: {test_output_path}")
            print("\nGeneration Info:")
            print(json.dumps(info, indent=2))
            # Clean up
            # os.remove(test_output_path)
            # print(f"Cleaned up {test_output_path}")
        else:
            print("\nTest image generation failed.")

    except (ConnectionError, TimeoutError, RuntimeError, ValueError) as e:
        print(f"\nTest failed: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred during testing: {e}")
