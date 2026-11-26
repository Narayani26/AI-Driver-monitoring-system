// Runs on the audio thread: forwards mono samples to the page in 0.25 s blocks.
// The AudioContext is created at 16 kHz, so no resampling is needed here.
class Forwarder extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buf = new Float32Array(4000);
    this.n = 0;
  }
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (ch) {
      for (let i = 0; i < ch.length; i++) {
        this.buf[this.n++] = ch[i];
        if (this.n === this.buf.length) {
          this.port.postMessage(this.buf.slice(0));
          this.n = 0;
        }
      }
    }
    return true;
  }
}
registerProcessor("forwarder", Forwarder);
