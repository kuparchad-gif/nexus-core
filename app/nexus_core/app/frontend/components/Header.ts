// components/Header.tsx - ADD VIREN STATUS
import { Badge, Group, ActionIcon, Tooltip } from '@mantine/core';
import { IconBrain, IconActivity, IconServer } from '@tabler/icons-react';

interface HeaderProps {
  user: User;
  onOpenSettings: () => void;
  onLock: () => void;
  onLogout: () => void;
}

const Header: React.FC<HeaderProps> = ({ user, onOpenSettings, onLock, onLogout }) => {
  const [virenStatus, setVirenStatus] = useState({
    health: 100,
    activeKubes: 0,
    systemLoad: 0
  });

  useEffect(() => {
    // Poll Viren status every 5 seconds
    const interval = setInterval(async () => {
      try {
        const response = await fetch('http://localhost:8080/health');
        const data = await response.json();
        setVirenStatus({
          health: Math.round(data.system_health * 100),
          activeKubes: data.kubes_active || 0,
          systemLoad: Math.round(psutil.cpu_percent())
        });
      } catch (error) {
        console.log('Viren monitoring offline');
      }
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  const getHealthColor = (health: number) => {
    if (health >= 80) return 'green';
    if (health >= 60) return 'yellow';
    return 'red';
  };

  return (
    <Group justify="space-between" px="md" py="sm" style={{ backdropFilter: 'blur(10px)' }}>
      <Group>
        <AetherealLogo width={40} height={40} />
        <Badge 
          leftSection={<IconBrain size={12} />}
          color={getHealthColor(virenStatus.health)}
          variant="light"
        >
          Viren: {virenStatus.health}%
        </Badge>
        <Badge 
          leftSection={<IconServer size={12} />}
          color="blue"
          variant="light"
        >
          Kubes: {virenStatus.activeKubes}
        </Badge>
      </Group>

      <Group>
        <Tooltip label="Viren System Monitor">
          <ActionIcon 
            variant="light" 
            onClick={() => {/* Open Viren monitor */}}
          >
            <IconActivity size={18} />
          </ActionIcon>
        </Tooltip>
        {/* Your existing buttons */}
      </Group>
    </Group>
  );
};