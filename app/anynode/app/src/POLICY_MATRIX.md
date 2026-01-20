# Policy Matrix (defaults)

| Action             | Condition                              | Decision                           |
|--------------------|----------------------------------------|------------------------------------|
| ping, heartbeat    | —                                      | allow (whitelist)                  |
| registry.read      | —                                      | allow (whitelist)                  |
| delete, wipe       | consent_token == CONSENT_TOKEN         | allow; else deny                   |
| seed.inject        | consent_token == CONSENT_TOKEN         | allow; else deny                   |
| fw.unlock          | consent_token == CONSENT_TOKEN         | allow; else deny                   |
| purchase, spend    | amount <= 200                          | allow; else deny                   |
| ascension.bypass   | meditation_minutes>=480 & metaphors>=12| allow; else deny                   |
| (other)            | —                                      | fallback: deny                     |
