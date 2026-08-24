const fs = require('fs');
const selfsigned = require('selfsigned');

(async () => {
    const attrs = [{ name: 'commonName', value: 'localhost' }];
    const pems = await selfsigned.generate(attrs, {
        days: 365,
        keySize: 2048,
        algorithm: 'sha256',
        extensions: [
            { name: 'subjectAltName', altNames: [
                { type: 2, value: 'localhost' },
                { type: 7, ip: '10.55.150.114' },
                { type: 7, ip: '127.0.0.1' },
            ]}
        ]
    });

    fs.writeFileSync('key.pem', pems.private);
    fs.writeFileSync('cert.pem', pems.cert);
    console.log('✅ Generated key.pem and cert.pem');
})();
