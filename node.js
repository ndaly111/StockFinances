// server.js
const express = require('express');
const fs = require('fs');
const path = require('path');
const bodyParser = require('body-parser');
const app = express();
const PORT = process.env.PORT || 3000;

app.use(bodyParser.json());

app.post('/manage_ticker', (req, res) => {
    const { action, ticker } = req.body;
    const filePath = action === 'add' ? 'add_ticker.csv' : 'remove_ticker.csv';
    fs.appendFile(path.join(__dirname, filePath), `${ticker}\n`, (err) => {
        if (err) {
            return res.status(500).json({ message: 'Failed to update CSV file.' });
        }
        res.status(200).json({ message: 'Update dispatched successfully.' });
    });
});

app.post('/update_growth', (req, res) => {
    const { ticker, growth_rate, profit_margin } = req.body;
    const data = `${ticker},${growth_rate},${profit_margin || ''}\n`;
    fs.appendFile(path.join(__dirname, 'update_growth.csv'), data, (err) => {
        if (err) {
            return res.status(500).json({ message: 'Failed to update CSV file.' });
        }
        res.status(200).json({ message: 'Update dispatched successfully.' });
    });
});

app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
});
