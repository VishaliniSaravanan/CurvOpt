/**
 * API Configuration
 * Automatically routes to the correct backend based on environment
 */

const API_BASE_URL = 
  import.meta.env.MODE === 'production'
    ? 'https://curvopt.onrender.com/api'  // Production: Render backend
    : '/api';  // Development: Vite proxy to localhost:5000

/**
 * Make API request with proper headers
 * @param {string} endpoint - API endpoint (e.g., 'optimize', 'status')
 * @param {object} options - Fetch options
 */
export const apiCall = async (endpoint, options = {}) => {
  const url = `${API_BASE_URL}/${endpoint}`;
  
  console.log(`[API] ${options.method || 'GET'} ${url}`);
  
  try {
    const response = await fetch(url, {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    });

    if (!response.ok) {
      throw new Error(`API Error: ${response.status} ${response.statusText}`);
    }

    return response;
  } catch (error) {
    console.error(`[API ERROR] ${endpoint}:`, error);
    throw error;
  }
};

/**
 * Stream API response (for Server-Sent Events)
 * @param {string} endpoint - API endpoint
 * @param {object} data - Request body
 * @param {function} onMessage - Callback for each message
 * @param {function} onError - Callback for errors
 */
export const streamApiCall = async (endpoint, data, onMessage, onError) => {
  const url = `${API_BASE_URL}/${endpoint}`;
  
  console.log(`[API STREAM] POST ${url}`);
  
  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(data),
    });

    if (!response.ok) {
      throw new Error(`API Error: ${response.status} ${response.statusText}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      
      buffer = lines.pop(); // Keep incomplete line in buffer

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          try {
            const data = JSON.parse(line.slice(6));
            onMessage(data);
          } catch (e) {
            console.warn('[API STREAM] Failed to parse message:', line);
          }
        }
      }
    }
  } catch (error) {
    console.error(`[API STREAM ERROR] ${endpoint}:`, error);
    if (onError) onError(error);
  }
};

export default {
  apiCall,
  streamApiCall,
  API_BASE_URL,
};
