const express = require('express');
const multer = require('multer');
const tf = require('@tensorflow/tfjs-node');
const { Storage } = require('@google-cloud/storage');
const { Firestore } = require('@google-cloud/firestore');
const { v4: uuidv4 } = require('uuid');
const path = require('path');
const fs = require('fs');

const app = express();
app.use(express.json());

const firestore = new Firestore();

const storage = new Storage();
const bucketName = 'predictions';

const upload = multer({
    dest: 'uploads/',
    limits: { fileSize: 1000000 },
});

let model;
const loadModel = async () => {
    const modelPath = `gs://${bucketName}/model/model.json`;
    model = await tf.loadGraphModel(modelPath);
};

loadModel().catch((error) => console.error('Error loading model:', error));

app.post('/predict', upload.single('image'), async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({ status: 'fail', message: 'No file uploaded' });
        }

        const imageBuffer = fs.readFileSync(req.file.path);
        const imageTensor = tf.node.decodeImage(imageBuffer)
            .resizeNearestNeighbor([224, 224])
            .toFloat()
            .expandDims();

        const prediction = await model.predict(imageTensor).data();
        const result = prediction[0] > 0.5 ? 'Cancer' : 'Non-cancer';
        const suggestion = result === 'Cancer' ? 'Segera periksa ke dokter!' : 'Penyakit kanker tidak terdeteksi.';
        const id = uuidv4();
        const createdAt = new Date().toISOString();

        // Save prediction result to Firestore
        await firestore.collection('predictions').doc(id).set({
            id, result, suggestion, createdAt,
        });

        res.json({
            status: 'success',
            message: 'Model is predicted successfully',
            data: { id, result, suggestion, createdAt },
        });

        // Clean up uploaded file
        fs.unlinkSync(req.file.path);
    } catch (error) {
        console.error('Prediction error:', error);
        res.status(400).json({ status: 'fail', message: 'Terjadi kesalahan dalam melakukan prediksi' });
    }
});

// Endpoint: History
app.get('/predict/histories', async (req, res) => {
    try {
        const snapshot = await firestore.collection('predictions').get();
        const histories = snapshot.docs.map((doc) => ({ id: doc.id, history: doc.data() }));

        res.json({ status: 'success', data: histories });
    } catch (error) {
        console.error('History error:', error);
        res.status(500).json({ status: 'fail', message: 'Gagal mengambil data riwayat prediksi' });
    }
});

const PORT = process.env.PORT || 8080;
app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
});