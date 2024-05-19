const tfjs = require('@tensorflow/tfjs-node');
const loadModel = () => {
    const modelUrl = "file://models/model.json";
    return tfjs.loadLayersModel(modelUrl);
};

const predict = (model, imageBuffer) => {
    const tensor = tfjs.node
        .decodeJpeg(imageBuffer)
        .resizeNearestNeighbor([150, 150])
        .expandDims()
        .toFloat();

    return model.predict(tensor).data();
};

module.exports = { loadModel, predict };