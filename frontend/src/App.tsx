import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Header from "./components/Header";
import Hero from "./components/Hero";
import About from "./components/About";
import DocumentsChatPage from "./components/DocumentsChatPage";
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
          <Route path="/rag" element={<DocumentsChatPage />} />
          <Route path="/metrics" element={<Metrics />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
