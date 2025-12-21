import React, { useRef } from "react";
import DocumentUpload, { DocumentUploadHandle } from "./DocumentUpload";
import DocumentList from "./DocumentList";
import Chat from "./Chat";
import "./DocumentsChatPage.css";

const DocumentsChatPage: React.FC = () => {
  const uploadRef = useRef<DocumentUploadHandle>(null);

  return (
    <div className="documents-chat-page">
      <div className="documents-chat-container">
        <div className="documents-section">
          <div className="documents-section-content">
            <h2 className="documents-section-title">Document Management</h2>
            <DocumentUpload ref={uploadRef} />
            <DocumentList />
          </div>
        </div>
        <div className="chat-section">
          <Chat />
        </div>
      </div>
    </div>
  );
};

export default DocumentsChatPage;
