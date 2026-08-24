require('dotenv').config();
const express = require('express');
const https = require('https');
const fs = require('fs');
const path = require('path');
const cors = require('cors');
const { createClient } = require('@supabase/supabase-js');

const app = express();

// Initialize Supabase client using Service Role Key to bypass RLS for uploads
const supabase = createClient(
    process.env.SUPABASE_URL,
    process.env.SUPABASE_SERVICE_ROLE_KEY
);
const BUCKET = process.env.SUPABASE_STORAGE_BUCKET;

// Increase payload limit for images
app.use(express.json({ limit: '50mb' }));
app.use(cors());

// Serve all static files from current directory
app.use(express.static(path.join(__dirname)));

// Endpoint to handle image uploads
app.post('/upload', async (req, res) => {
    try {
        const { imageBase64 } = req.body;
        if (!imageBase64) {
            return res.status(400).json({ error: 'No image data provided' });
        }

        // Remove the data URI header "data:image/png;base64,"
        const base64Data = imageBase64.replace(/^data:image\/png;base64,/, "");
        const buffer = Buffer.from(base64Data, 'base64');
        
        const filename = `MemeMatch_${Date.now()}.png`;

        // Upload to Supabase Storage
        const { data, error } = await supabase
            .storage
            .from(BUCKET)
            .upload(filename, buffer, {
                contentType: 'image/png',
                upsert: false
            });

        if (error) {
            console.error("Supabase upload error:", error);
            return res.status(500).json({ error: 'Failed to upload to Supabase' });
        }

        // Get the public URL for the uploaded file
        const { data: publicUrlData } = supabase
            .storage
            .from(BUCKET)
            .getPublicUrl(filename);

        console.log(`Saved snapshot to Supabase: ${publicUrlData.publicUrl}`);
        res.json({ success: true, url: publicUrlData.publicUrl });
        
    } catch (err) {
        console.error("Upload error:", err);
        res.status(500).json({ error: 'Internal server error' });
    }
});

const PORT = process.env.PORT || 8080;

try {
    const options = {
        key: fs.readFileSync('key.pem'),
        cert: fs.readFileSync('cert.pem')
    };
    https.createServer(options, app).listen(PORT, '0.0.0.0', () => {
        console.log(`===========================================`);
        console.log(`🚀 MemeMatch Server running with Auto-Save!`);
        console.log(`Local  : https://localhost:${PORT}`);
        console.log(`===========================================`);
    });
} catch (e) {
    console.warn("⚠️ No SSL certificates found. Falling back to standard HTTP.");
    console.warn("This is normal if you are deploying to a cloud service (Render, Railway, etc.) that handles HTTPS for you.");
    app.listen(PORT, '0.0.0.0', () => {
        console.log(`===========================================`);
        console.log(`🚀 MemeMatch Server running with Auto-Save!`);
        console.log(`Network: http://0.0.0.0:${PORT}`);
        console.log(`===========================================`);
    });
}
