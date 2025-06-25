export class MockWebSocket {
  private ws: WebSocket | null = null;
  private listeners: { [key: string]: ((event: MessageEvent) => void)[] } = {};
  private _readyState: number = WebSocket.CONNECTING;

  constructor(url: string) {
    console.log('Mock WebSocket connecting to:', url);
    
    // Simulate connection delay
    setTimeout(() => {
      this._readyState = WebSocket.OPEN;
      console.log('Mock WebSocket connected');
      
      // Trigger onopen if it exists
      if (this.onopen) {
        this.onopen(new Event('open'));
      }
    }, 500);
  }

  onopen: ((event: Event) => void) | null = null;
  onerror: ((event: Event) => void) | null = null;
  onmessage: ((event: MessageEvent) => void) | null = null;

  get readyState() {
    return this._readyState;
  }

  send(data: string) {
    console.log('Mock WebSocket sending:', data);
    
    try {
      const message = JSON.parse(data);
      
      // Simulate different responses based on message type
      if (message.type === 'video_upload') {
        this.simulateVideoUploadResponse();
      } else if (message.type === 'annotation_update') {
        this.simulateInferenceResponse(message);
      }
    } catch (error) {
      console.error('Error parsing WebSocket message:', error);
    }
  }

  addEventListener(type: string, listener: (event: MessageEvent) => void) {
    if (!this.listeners[type]) {
      this.listeners[type] = [];
    }
    this.listeners[type].push(listener);
  }

  removeEventListener(type: string, listener: (event: MessageEvent) => void) {
    if (this.listeners[type]) {
      this.listeners[type] = this.listeners[type].filter(l => l !== listener);
    }
  }

  close() {
    this._readyState = WebSocket.CLOSED;
    console.log('Mock WebSocket closed');
  }

  private simulateVideoUploadResponse() {
    setTimeout(() => {
      const response = {
        type: 'video_upload_success',
        message: 'Video processed successfully',
        timestamp: new Date().toISOString()
      };
      this.sendMessage(response);
    }, 1000);
  }

  private simulateInferenceResponse(annotationMessage: any) {
    // Send inference started
    setTimeout(() => {
      const startResponse = {
        type: 'inference_started',
        timestamp: new Date().toISOString()
      };
      this.sendMessage(startResponse);
    }, 100);

    // Send inference result with mock mask data
    setTimeout(() => {
      const mockMaskData = this.generateMockMaskImage();
      const resultResponse = {
        type: 'inference_result',
        mask_data: mockMaskData,
        objects: annotationMessage.objects,
        timestamp: new Date().toISOString()
      };
      this.sendMessage(resultResponse);
    }, 2000);
  }

  private generateMockMaskImage(): string {
    // Generate a simple base64 encoded image (a colored rectangle as mock mask)
    const canvas = document.createElement('canvas');
    canvas.width = 640;
    canvas.height = 480;
    const ctx = canvas.getContext('2d');
    
    if (ctx) {
      // Create a gradient mask effect
      const gradient = ctx.createRadialGradient(320, 240, 0, 320, 240, 200);
      gradient.addColorStop(0, 'rgba(59, 130, 246, 0.8)'); // Blue center
      gradient.addColorStop(1, 'rgba(59, 130, 246, 0.2)'); // Faded edges
      
      ctx.fillStyle = gradient;
      ctx.fillRect(0, 0, 640, 480);
      
      // Add some random shapes to simulate segmentation
      ctx.fillStyle = 'rgba(34, 197, 94, 0.6)'; // Green
      ctx.beginPath();
      ctx.ellipse(200, 150, 80, 60, 0, 0, 2 * Math.PI);
      ctx.fill();
      
      ctx.fillStyle = 'rgba(239, 68, 68, 0.6)'; // Red
      ctx.beginPath();
      ctx.ellipse(450, 300, 60, 40, 0, 0, 2 * Math.PI);
      ctx.fill();
    }
    
    // Convert to base64 (remove the data:image/png;base64, prefix)
    return canvas.toDataURL('image/png').split(',')[1];
  }

  private sendMessage(data: any) {
    const event = new MessageEvent('message', {
      data: JSON.stringify(data)
    });
    
    if (this.onmessage) {
      this.onmessage(event);
    }
    
    // Also trigger addEventListener listeners
    if (this.listeners['message']) {
      this.listeners['message'].forEach(listener => listener(event));
    }
  }
}

// Override the global WebSocket constructor for our mock
export const setupMockWebSocket = () => {
  (window as any).OriginalWebSocket = window.WebSocket;
  (window as any).WebSocket = MockWebSocket;
};

export const restoreWebSocket = () => {
  if ((window as any).OriginalWebSocket) {
    (window as any).WebSocket = (window as any).OriginalWebSocket;
  }
};
