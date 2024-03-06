const HMKit = require('hmkit')

const hmkit = new HMKit(
    "c2JveHfHT3d3Y0B+l7Uhzgj04TpOicNtD0ArBOxgndHlGKkZNXx35Dj0/Tltsaf1i3EfI+snTbsl16q+g8S1d6f8nBdcmiQIqOPg2/aGNTJcpt9mnrDJk53UGgNdCUFYYh6WwSEaoMYcN2H/jviEZqE3YBAF4xv4VB7Ezedq6h9DsBfD5Ofw3uTYImVeNL9hm5gE/brEyam1",
    "rUVXGmKdnxvpTOlMSX4Bgyy/tLyiQ6dMWTqhq+B7DKg="
);

async function app() {

    await fetch('https://sandbox.owner-panel.high-mobility.com/oauth/new', {
        method: 'GET',
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({
           client_id: '4a2680bf-1313-4032-8906-927037218d47',
           redirect_uri: 'https://getpostman.com/oauth2/callback'
        }),
    })
    .then((response) => response.json())
    .then((data) => console.log(data))
    console.log("hi")

    await fetch('https://sandbox.api.high-mobility.com/v1/access_tokens', {
    method: "POST",
    headers: {
        "Content-Type": "application/json",
    },
    body: JSON.stringify({
        code:'487470e0-621d-42d8-b1f6-1db6c6ded545',
        client_id: '4a2680bf-1313-4032-8906-927037218d47',
        grant_type: 'authorization_code',
        redirect_uri: 'https://getpostman.com/oauth2/callback'
      }),
    })
    .then((response) => response.json())
    .then((data) => console.log(data));

    const accessCertificate = await hmkit.downloadAccessCertificate(
      "4a2680bf-1313-4032-8906-927037218d47"
    );
  
    try {
      const response = await hmkit.telematics.sendCommand(
        hmkit.commands.Ignition.turnIgnitionOnOff({
          state: "on" // Available values: ['on', 'off', 'start', 'lock', 'accessory']
        }),
        accessCertificate
      );
  
      console.log(response.bytes());
      console.log(response.parse());
    } catch (e) {
      console.log(e);
    }
  }
  
  // Run your app
app();