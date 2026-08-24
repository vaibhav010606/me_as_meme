const { createClient } = require('@supabase/supabase-js');

const handler = async function(req, res) {
    if (!process.env.SUPABASE_URL || !process.env.SUPABASE_SERVICE_ROLE_KEY) {
        console.error("Missing Supabase Environment Variables!");
        return res.status(500).json({ error: 'Server misconfiguration: Missing Supabase keys.' });
    }

    const supabase = createClient(
        process.env.SUPABASE_URL,
        process.env.SUPABASE_SERVICE_ROLE_KEY
    );
    const BUCKET = process.env.SUPABASE_STORAGE_BUCKET || 'memory-photos';
    if (req.method !== 'POST') {
        return res.status(405).json({ error: 'Method Not Allowed' });
    }

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
}

handler.config = {
  api: {
    bodyParser: {
      sizeLimit: '50mb',
    },
  },
};

module.exports = handler;
