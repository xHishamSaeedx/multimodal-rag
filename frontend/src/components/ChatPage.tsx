import React from 'react';
import Chat from './Chat';
import './ChatPage.css';

const ChatPage: React.FC = () => {
  return (
    <div className="chat-page">
      <div className="chat-page-container">
        <Chat />
      </div>
    </div>
  );
};

export default ChatPage;
