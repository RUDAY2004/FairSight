
 const firebaseConfig = {
    apiKey: "AIzaSyCVxozAlc5WMhY9_XRxy_nd2E8io_-fUqY",
    authDomain: "fairsightauth.firebaseapp.com",
    projectId: "fairsightauth",
    storageBucket: "fairsightauth.appspot.com",
    messagingSenderId: "440786177943",
    appId: "1:440786177943:web:3317433cfc3aa9d6b83ed2",
    measurementId: "G-MQ1E3KMWGK"
  };
  firebase.initializeApp(firebaseConfig);
  const auth = firebase.auth();
const db = firebase.firestore();

window.signup = async function(event) {
  event.preventDefault();

  const name = document.getElementById('name').value.trim();
  const phone = document.getElementById('phone').value.trim();
  const email = document.getElementById('email').value.trim();
  const password = document.getElementById('password').value.trim();
  const confirmPassword = document.getElementById('confirmPassword').value.trim();
  const company = document.getElementById('company').value.trim();
  const domain = document.getElementById('domains').value;

  if (password !== confirmPassword) {
    alert("Passwords do not match!");
    return;
  }

  try {
    const userCredential = await auth.createUserWithEmailAndPassword(email, password);
    const user = userCredential.user;

    await db.collection("users").doc(user.uid).set({
      uid: user.uid,
      name,
      phone,
      email,
      company,
      domain,
      createdAt: firebase.firestore.FieldValue.serverTimestamp()
    });

    alert("Signup successful! Redirecting to login...");
    window.location.href = "login.html";

  } catch (error) {
    console.error("Signup error:", error);
    alert("Signup failed: " + error.message);
  }
};

window.login = async function(event) {
  event.preventDefault();

  const email = document.getElementById('email').value.trim();
  const password = document.getElementById('password').value.trim();

  try {
    const userCredential = await auth.signInWithEmailAndPassword(email, password);
    const user = userCredential.user;

    // Log the login time
    await db.collection("users").doc(user.uid).collection("logins").add({
      email: user.email,
      loginTime: firebase.firestore.FieldValue.serverTimestamp()
    });

    // Fetch the user's domain from Firestore
    const userDoc = await db.collection("users").doc(user.uid).get();
    if (userDoc.exists) {
      const domain = userDoc.data().domain.toLowerCase(); // e.g., "hr", "banking"
      const redirectURL = `dashboard_${domain}.html`;

      alert("Login successful! Redirecting...");
      window.location.href = redirectURL;
    } else {
      alert("User data not found. Please contact support.");
    }

  } catch (error) {
    console.error("Login error:", error);
    if (error.code === 'auth/user-not-found') {
      alert("❌ User not registered. Please sign up first.");
    } else if (error.code === 'auth/wrong-password') {
      alert("⚠️ Incorrect password. Try again.");
    } else {
      alert("Login failed: " + error.message);
    }
  }
};
