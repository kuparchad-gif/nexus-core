import React, { useState, useRef, useEffect } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, Dimensions } from 'react-native';
import Video from 'react-native-video';

const App = () => {
  const [activePage, setActivePage] = useState('home');
  const [isRecording, setIsRecording] = useState(false);
  const videoRef = useRef(null);

  useEffect(() => {
    if (activePage === 'voice' && videoRef.current) {
      videoRef.current.play();
    } else if (videoRef.current) {
      videoRef.current.pause();
      videoRef.current.seek(0);
    }
  }, [activePage]);

  const toggleRecording = () => setIsRecording(!isRecording);
  const navigate = (page) => setActivePage(page);

  const renderContent = () => {
    switch (activePage) {
      case 'home':
        return (
          <View style={styles.homeContent}>
            <Text style={styles.title}>Aethereal Nexus</Text>
            <View style={styles.gaugeContainer}>
              {['Lilith: 94%', 'VIREN: 88%'].map((label, i) => (
                <View style={styles.gaugeCard} key={i}>
                  <Text>{label}</Text>
                  <View style={styles.gaugeBar}>
                    <View style={[styles.gaugeFill, { width: label.split(': ')[1] }]} />
                  </View>
                </View>
              ))}
            </View>
            <TouchableOpacity style={styles.button} onPress={() => navigate('chat')}>
              <Text>Enter Chat</Text>
            </TouchableOpacity>
          </View>
        );
      case 'voice':
        return (
          <View style={styles.voiceContent}>
            <Video
              ref={videoRef}
              source={require('./assets/morph_orb.mp4')}
              style={styles.backgroundVideo}
              resizeMode="cover"
              repeat={true}
              muted={true}
            />
            <View style={styles.voiceControls}>
              <TouchableOpacity style={styles.button} onPress={() => navigate('home')}>
                <Text>Back</Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={[styles.button, isRecording && styles.recording]}
                onPress={toggleRecording}
              >
                <Text>{isRecording ? 'Mute' : 'Record'}</Text>
              </TouchableOpacity>
            </View>
            {isRecording && <Text style={styles.recordingText}>Listening...</Text>}
          </View>
        );
      default:
        return null;
    }
  };

  return (
    <View style={styles.container}>
      <View style={styles.sidebar}>
        {['home', 'voice'].map((page) => (
          <TouchableOpacity
            style={[styles.navItem, activePage === page && styles.active]}
            onPress={() => navigate(page)}
            key={page}
          >
            <Text>{page.charAt(0).toUpperCase() + page.slice(1)}</Text>
          </TouchableOpacity>
        ))}
      </View>
      <View style={styles.mainContent}>{renderContent()}</View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1, flexDirection: 'row' },
  sidebar: { width: 100, backgroundColor: 'rgba(255, 255, 255, 0.2)', padding: 10 },
  navItem: { padding: 10, borderRadius: 5 },
  active: { backgroundColor: 'rgba(0, 255, 255, 0.4)' },
  mainContent: { flex: 1, justifyContent: 'center', alignItems: 'center' },
  homeContent: { alignItems: 'center' },
  title: { fontSize: 24, marginBottom: 20 },
  gaugeContainer: { flexDirection: 'row', gap: 10 },
  gaugeCard: { padding: 10 },
  gaugeBar: { height: 10, backgroundColor: 'rgba(255, 255, 255, 0.1)', borderRadius: 5 },
  gaugeFill: { height: '100%', backgroundColor: '#00f6ff' },
  button: { padding: 10, backgroundColor: 'rgba(255, 255, 255, 0.1)', borderRadius: 5 },
  voiceContent: { width: '100%', height: '100%' },
  backgroundVideo: { position: 'absolute', top: 0, left: 0, width: '100%', height: '100%' },
  voiceControls: { flexDirection: 'row', gap: 10, position: 'absolute', bottom: 20 },
  recording: { backgroundColor: 'rgba(255, 0, 77, 0.3)' },
  recordingText: { color: '#ff004d', position: 'absolute', bottom: 50 },
});

export default App;