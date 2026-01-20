var dagcomponentfuncs = window.dashAgGridComponentFunctions = window.dashAgGridComponentFunctions || {};

dagcomponentfuncs.ModelLink = function(props) {
    if (!props.data.Model_Link) {
        return props.value;  // Just return text if no link
    }
    return React.createElement(
        'a',
        {
            href: props.data.Model_Link,
            target: '_blank',
            style: {
                color: '#007bff',
                textDecoration: 'none'
            }
        },
        props.value
    );
};

dagcomponentfuncs.PinRenderer = function(props) {
    return React.createElement(
        'div',
        {
            onClick: function() {
                const api = props.api;
                const modelId = props.data.Model_Display;
                const isPinned = props.data.pinned || false;
                
                if (isPinned) {
                    // Unpin
                    const currentPinned = api.getGridOption('pinnedTopRowData') || [];
                    const newPinnedRows = currentPinned.filter(row => row.Model_Display !== modelId);
                    api.setGridOption('pinnedTopRowData', newPinnedRows);
                } else {
                    // Pin
                    const currentPinned = api.getGridOption('pinnedTopRowData') || [];
                    const pinnedRow = {...props.data, pinned: true};
                    api.setGridOption('pinnedTopRowData', [...currentPinned, pinnedRow]);
                }
            },
            style: {
                cursor: 'pointer',
                textAlign: 'center',
                fontSize: '16px'
            }
        },
        props.data.pinned ? '📌' : '☐'
    );
};

dagcomponentfuncs.TypeRenderer = function(props) {
    const typeMap = {
        'Base': ['B', '#71de5f'],
        'Finetune': ['F', '#f6b10b'], 
        'Merge': ['M', '#f08aff'],
        'Proprietary': ['P', '#19cdce']
    };
    
    // Determine type from raw flags
    let type = 'Unknown';
    if (props.data['Total Parameters'] === null) {
        type = 'Proprietary';
    } else if (props.data['Is Foundation'] && !props.data['Is Merged']) {
        type = 'Base';
    } else if (props.data['Is Merged']) {
        type = 'Merge';
    } else if (props.data['Is Finetuned'] && !props.data['Is Merged']) {
        type = 'Finetune';
    }
    
    const [letter, color] = typeMap[type] || ['?', '#999'];
    
    return React.createElement('div', {
        style: {
            display: 'flex',
            alignItems: 'center',
            height: '100%',
            position: 'absolute',
            top: 0,
            bottom: 0,
            left: '12px'
        }
    }, 
        React.createElement('div', {
            style: {
                color: color,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontWeight: 'bold',
                fontSize: '14px',
                lineHeight: '1',
                textAlign: 'center'
            }
        }, letter)
    );
};
