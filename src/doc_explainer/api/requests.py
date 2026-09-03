from typing import Dict


class Request:
    pass

class Response:
    pass

class RequestHandler:
    def send_request(self, endpoint: str, req: Request) -> Dict:
        # Simulate sending a request to an external service
        print(f"Sending request to {endpoint} with data: {req}")
        # Simulated response
        return {"status": "success", "data": "Response from " + endpoint}
    def receive_response(self, response: Dict) -> None:
        # Simulate processing the received response
        print(f"Received response: {response}")
        # Here you would typically update your application state based on the response
    def process_request(self, endpoint: str, data: Dict) -> None:
        response = self.send_request(endpoint, data)
        self.receive_response(response) 