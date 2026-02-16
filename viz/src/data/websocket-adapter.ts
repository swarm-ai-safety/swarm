import type { EpochSnapshot, AgentSnapshot } from "./types";

/** Callback for live data events */
export interface LiveDataCallbacks {
  onEpoch: (epoch: EpochSnapshot) => void;
  onAgents: (agents: AgentSnapshot[]) => void;
  onError: (error: Error) => void;
  onConnect: () => void;
  onDisconnect: () => void;
}

/**
 * WebSocket adapter stub for future real-time simulation monitoring.
 * Not yet implemented - provides the interface contract.
 */
export class WebSocketAdapter {
  private ws: WebSocket | null = null;
  private callbacks: LiveDataCallbacks;
  private url: string;

  constructor(url: string, callbacks: LiveDataCallbacks) {
    this.url = url;
    this.callbacks = callbacks;
  }

  connect() {
    try {
      this.ws = new WebSocket(this.url);
      this.ws.onopen = () => this.callbacks.onConnect();
      this.ws.onclose = () => this.callbacks.onDisconnect();
      this.ws.onerror = (e) => this.callbacks.onError(new Error("WebSocket error"));
      this.ws.onmessage = (evt) => this.handleMessage(evt.data);
    } catch (e) {
      this.callbacks.onError(e instanceof Error ? e : new Error(String(e)));
    }
  }

  disconnect() {
    this.ws?.close();
    this.ws = null;
  }

  private handleMessage(data: string) {
    try {
      const msg = JSON.parse(data);
      if (msg.type === "epoch") {
        this.callbacks.onEpoch(msg.data as EpochSnapshot);
      } else if (msg.type === "agents") {
        this.callbacks.onAgents(msg.data as AgentSnapshot[]);
      }
    } catch (e) {
      this.callbacks.onError(e instanceof Error ? e : new Error(String(e)));
    }
  }
}
