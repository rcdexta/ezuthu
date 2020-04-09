import React from "react";
import "./styles.css";
import CanvasDraw from "react-canvas-draw";
import axios from "axios";
import { Container, Row, Col } from "react-grid-system";

const dataURLtoFile = (dataurl, filename) => {
  const arr = dataurl.split(",");
  const mime = arr[0].match(/:(.*?);/)[1];
  const bstr = atob(arr[1]);
  let n = bstr.length;
  const u8arr = new Uint8Array(n);
  while (n) {
    u8arr[n - 1] = bstr.charCodeAt(n - 1);
    n -= 1; // to make eslint happy
  }
  return new File([u8arr], filename, { type: mime });
};

export default class App extends React.Component {
  state = { ranks: undefined };

  constructor(props) {
    super(props);
    this.saveableCanvas = React.createRef();
  }

  clear = () => {
    this.saveableCanvas.clear();
    this.setState({ ranks: undefined });
  };

  saveImage = () => {
    const canvas = this.saveableCanvas.canvas.drawing;
    const backgroundColor = "#fff";
    const type = "png";
    let context = canvas.getContext("2d");
    //cache height and width
    let w = canvas.width;
    let h = canvas.height;
    let data;

    if (backgroundColor) {
      //get the current ImageData for the canvas.
      data = context.getImageData(0, 0, w, h);

      //store the current globalCompositeOperation
      var compositeOperation = context.globalCompositeOperation;

      //set to draw behind current content
      context.globalCompositeOperation = "destination-over";

      //set background color
      context.fillStyle = backgroundColor;

      //draw background / rect on entire canvas
      context.fillRect(0, 0, w, h);
    }

    //get the image data from the canvas
    var imageData = canvas.toDataURL(`image/${type}`);

    if (backgroundColor) {
      //clear the canvas
      context.clearRect(0, 0, w, h);

      //restore it with original / cached ImageData
      context.putImageData(data, 0, 0);

      //reset the globalCompositeOperation to what it was
      context.globalCompositeOperation = compositeOperation;
    }

    const file = dataURLtoFile(imageData);

    const formData = new FormData();
    formData.append("file", file, file.name);

    // now upload
    const config = {
      headers: { "Content-Type": "multipart/form-data" },
    };
    axios.post("/upload", formData, config).then((response) => {
      const results = response.data;
      this.setState({ ranks: results });
      window.scrollTo(0, document.body.scrollHeight);
    });
  };

  render() {
    const { ranks } = this.state;
    return (
      <div className="App">
        <Container>
          <Row>
            <Col>
              <h2>Ezuthu - Tamil Character Classifier</h2>
            </Col>
          </Row>
        </Container>
        <Container>
          <Row>
            <Col sm={6}>
              Draw a tamil alphabet filling the entire space as big as you can:
              <div style={{ border: "1px solid #000", marginTop: 5 }}>
                <CanvasDraw
                  ref={(canvasDraw) => (this.saveableCanvas = canvasDraw)}
                  lazyRadius={0}
                  brushColor="#000"
                  canvasWidth="100%"
                />
              </div>
              <button onClick={() => window.open('https://github.com/rcdexta/ezuthu#what-is-this-project')}>Help</button>
              <button onClick={this.clear}>Clear</button>
              <button
                onClick={() => {
                  this.saveableCanvas.undo();
                }}
              >
                Undo
              </button>
              <button onClick={this.saveImage}>Submit</button>
            </Col>
            <Col sm={6}>
              <Container>
                <Row>
                  <Col>
                    {ranks && (
                      <div style={{ margin: "0.5em" }}>
                        <div style={{ fontSize: "4em" }}>{ranks.rank1}</div>
                        <div style={{ fontSize: 11, marginTop: 3 }}>
                          Guess 1
                        </div>
                      </div>
                    )}
                  </Col>
                </Row>
                <Row>
                  <Col>
                    {ranks && (
                      <div style={{ margin: "0.5em" }}>
                        <div style={{ fontSize: "4em" }}>{ranks.rank2}</div>
                        <div style={{ fontSize: 11, marginTop: 3 }}>
                          Guess 2
                        </div>
                      </div>
                    )}
                  </Col>
                </Row>
                <Row>
                  <Col>
                    {ranks && (
                      <div style={{ margin: "0.5em" }}>
                        <div style={{ fontSize: "4em" }}>{ranks.rank3}</div>
                        <div style={{ fontSize: 11, marginTop: 3 }}>
                          Guess 3
                        </div>
                      </div>
                    )}
                  </Col>
                </Row>
              </Container>
            </Col>
          </Row>
        </Container>
      </div>
    );
  }
}
