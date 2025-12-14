import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Header from "./components/Header";
import Hero from "./components/Hero";
import About from "./components/About";
import ChatPage from "./components/ChatPage";
import Documents from "./components/Documents";
import Metrics from "./components/Metrics";
import "./App.css";

function HomePage() {
  return (
    <>
      <Hero />
      <About />
    </>
  );
}

function App() {
  return (
    <Router>
      <div className="App">
        <Header />
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/documents" element={<Documents />} />
          <Route path="/chat" element={<ChatPage />} />
          <Route path="/metrics" element={<Metrics />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
