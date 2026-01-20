import React, { StrictMode, useEffect, useState } from "https://esm.sh/react";
import { createRoot } from "https://esm.sh/react-dom/client";
import { Flipper, Flipped } from "https://esm.sh/react-flip-toolkit";
import { faker } from "https://esm.sh/@faker-js/faker";
createRoot(document.getElementById("root")).render(React.createElement(StrictMode, null,
    React.createElement(IconSprites, null),
    React.createElement("main", null,
        React.createElement(FileBrowser, null))));
/**
 * Generate fake files and folders.
 * @param count Number of items to generate
 */
function generateFakeItems(count) {
    const items = [];
    for (let c = 0; c < count; ++c) {
        const shouldBeDirectory = faker.number.float() < 0.5;
        const users = faker.number.int({ max: 5 });
        const id = faker.string.uuid();
        const sharing = [];
        const modified = faker.date.past({ years: 2 });
        for (let u = 0; u < users; ++u) {
            // users with whom items will be shared
            const sex = faker.person.sex();
            sharing.push({
                name: faker.person.fullName({ sex }),
                avatar: faker.image.personPortrait({ sex, size: 64 })
            });
        }
        if (shouldBeDirectory) {
            // folder
            items.push({
                id,
                name: faker.system.commonFileName().split(".")[0],
                sharing,
                size: faker.number.int({ min: 1e4, max: 3e6 }),
                modified,
                isDirectory: true
            });
        }
        else {
            // file
            const fileTypes = ["html", "md", "txt"];
            const fileTypeIndex = faker.number.int({ max: fileTypes.length - 1 });
            const fileType = fileTypes[fileTypeIndex];
            items.push({
                id,
                name: faker.system.commonFileName(fileType),
                sharing,
                size: faker.number.int({ max: 1e6 }),
                modified
            });
        }
    }
    return items;
}
function FileBrowser() {
    const [pristine, setPristine] = useState(true);
    const [files, setFiles] = useState(generateFakeItems(5));
    const [sortBy, setSortBy] = useState("name");
    const [sortDir, setSortDir] = useState("ascending");
    const sortOptions = [
        { label: "Name", property: "name" },
        { label: "Sharing", property: "sharing" },
        { label: "Size", property: "size" },
        { label: "Modified", property: "modified" },
    ];
    const sortFunctions = {
        name: (a, b, dir) => Utils.sortStrings(a.name, b.name, dir),
        sharing: (a, b, dir) => Utils.sortNumbers(a.sharing.length, b.sharing.length, dir),
        size: (a, b, dir) => Utils.sortNumbers(a.size, b.size, dir),
        modified: (a, b, dir) => Utils.sortDates(a.modified, b.modified, dir)
    };
    const sortSetter = (property) => {
        // allow FLIP transitions to run
        setPristine(false);
        if (sortBy === property) {
            setSortDir(sortDir === "ascending" ? "descending" : "ascending");
            return;
        }
        setSortBy(property);
    };
    useEffect(() => {
        const sorted = [...files].sort((a, b) => sortFunctions[sortBy](a, b, sortDir));
        setFiles(sorted);
    }, [sortDir, sortBy]);
    return (React.createElement("div", { className: "browser", role: "grid", "aria-rowcount": files.length + 1 },
        React.createElement("div", { role: "rowgroup" },
            React.createElement(FileBrowserHeader, null,
                sortOptions.map((option, i) => (React.createElement(FileBrowserHeaderSort, Object.assign({ key: i }, option, { sortDir: sortBy === option.property ? sortDir : "none", sortSetter: () => sortSetter(option.property) })))),
                React.createElement("div", { className: "browser__header-cell", role: "columnheader", "aria-sort": "none" },
                    React.createElement("span", { className: "sr-only" }, "More actions")))),
        React.createElement("div", { className: "browser__line" }),
        React.createElement("div", { role: "rowgroup" },
            React.createElement(Flipper, { flipKey: pristine ? "initial" : files.map(file => file.id).join("") }, files.map((file, i) => (React.createElement(Flipped, { key: file.id, flipId: file.id },
                React.createElement("div", { className: "browser__row", role: "row", "aria-rowindex": i + 2 },
                    React.createElement(FileBrowserRow, Object.assign({ key: i }, file))))))))));
}
function FileBrowserAvatar({ name, avatar }) {
    return (React.createElement("div", { className: "avatar" },
        React.createElement("img", { className: "avatar__img", src: avatar, alt: name, title: name, width: "24", height: "24" })));
}
function FileBrowserHeader({ children }) {
    return (React.createElement("div", { className: "browser__header", role: "row", "aria-rowindex": 1 }, children));
}
function FileBrowserHeaderSort({ label, sortDir, sortSetter }) {
    const icons = {
        ascending: "caret-up",
        descending: "caret-down",
        none: ""
    };
    const ariaLabel = {
        ascending: "(Ascending)",
        descending: "(Descending)",
        none: ""
    };
    const current = sortDir !== "none";
    const buttonClass = "browser__col-button";
    const buttonClasses = `${buttonClass}${current ? ` ${buttonClass}--active` : ""}`;
    return (React.createElement("div", { className: "browser__header-cell", role: "columnheader", "aria-sort": sortDir },
        React.createElement("button", { className: buttonClasses, type: "button", onClick: sortSetter },
            label,
            current && React.createElement("span", { "aria-label": ariaLabel[sortDir] },
                React.createElement(Icon, { icon: icons[sortDir] })))));
}
function FileBrowserRow({ name, sharing, size, modified, isDirectory }) {
    const icon = isDirectory === true ? "folder" : `file-${name.split(".").pop()}`;
    const isPrivate = sharing.length > 0;
    const avatarLimit = 3;
    const sharingFirst = sharing.slice(0, avatarLimit);
    const sharingRest = Math.max(0, sharing.length - avatarLimit);
    const sharingStatus = isPrivate ? `${sharing.length} user(s)` : "Public";
    const sizeString = Utils.formatBytes(size, 0);
    const modifiedString = Utils.formatDate(modified);
    return (React.createElement(React.Fragment, null,
        React.createElement("div", { className: "browser__row-cell", role: "gridcell" },
            React.createElement(Icon, { icon: icon }),
            React.createElement("div", { className: "browser__row-item" },
                React.createElement("div", { className: "browser__row-name", title: name }, name),
                React.createElement("div", { className: "browser__row-details" },
                    sharingStatus,
                    " \u2022 ",
                    sizeString,
                    " \u2022 ",
                    modifiedString))),
        React.createElement("div", { className: "browser__row-cell", role: "gridcell" }, isPrivate ? React.createElement(React.Fragment, null,
            sharingFirst.map((user, i) => React.createElement(FileBrowserAvatar, Object.assign({ key: i }, user))),
            sharingRest > 0 && React.createElement("span", { className: "avatar__count" },
                "+",
                sharingRest)) : sharingStatus),
        React.createElement("div", { className: "browser__row-cell", role: "gridcell" }, sizeString),
        React.createElement("div", { className: "browser__row-cell", role: "gridcell" }, modifiedString),
        React.createElement("div", { className: "browser__row-cell", role: "gridcell" },
            React.createElement("button", { className: "browser__row-button", type: "button", title: "More actions" },
                React.createElement(Icon, { icon: "more" })))));
}
function Icon({ icon }) {
    const href = `#${icon}`;
    return (React.createElement("svg", { className: "icon", width: "16px", height: "16px", "aria-hidden": "true" },
        React.createElement("use", { href: href })));
}
function IconSprites() {
    const viewBox = "0 0 16 16";
    return (React.createElement("svg", { width: "0", height: "0", display: "none" },
        React.createElement("symbol", { id: "caret-down", viewBox: viewBox },
            React.createElement("polyline", { fill: "none", stroke: "currentcolor", strokeLinecap: "round", strokeLinejoin: "round", strokeWidth: "2", points: "4 6,8 10,12 6" })),
        React.createElement("symbol", { id: "caret-up", viewBox: viewBox },
            React.createElement("polyline", { fill: "none", stroke: "currentcolor", strokeLinecap: "round", strokeLinejoin: "round", strokeWidth: "2", points: "4 10,8 6,12 10" })),
        React.createElement("symbol", { id: "file-html", viewBox: viewBox },
            React.createElement("polygon", { fill: "hsl(270,90%,90%)", points: "1.5 0,9.5 0,14.5 5,14.5 16,1.5 16" }),
            React.createElement("polygon", { fill: "hsl(270,90%,50%)", points: "9.5 0,14.5 5,9.5 5" }),
            React.createElement("g", { fill: "none", stroke: "hsl(270,90%,50%)", strokeWidth: "1" },
                React.createElement("polyline", { points: "7 13,9 8" }),
                React.createElement("polyline", { points: "5.5 9,4 10.5,5.5 12" }),
                React.createElement("polyline", { points: "10.5 9,12 10.5,10.5 12" }))),
        React.createElement("symbol", { id: "file-md", viewBox: viewBox },
            React.createElement("polygon", { fill: "hsl(0,0%,80%)", points: "1.5 0,9.5 0,14.5 5,14.5 16,1.5 16" }),
            React.createElement("polygon", { fill: "hsl(0,0%,40%)", points: "9.5 0,14.5 5,9.5 5" })),
        React.createElement("symbol", { id: "file-txt", viewBox: viewBox },
            React.createElement("polygon", { fill: "hsl(210,90%,80%)", points: "1.5 0,9.5 0,14.5 5,14.5 16,1.5 16" }),
            React.createElement("polygon", { fill: "hsl(210,90%,40%)", points: "9.5 0,14.5 5,9.5 5" }),
            React.createElement("g", { stroke: "hsl(210,90%,40%)", strokeWidth: "2" },
                React.createElement("polyline", { points: "4 7,10 7" }),
                React.createElement("polyline", { points: "4 10,8 10" }),
                React.createElement("polyline", { points: "4 13,12 13" }))),
        React.createElement("symbol", { id: "folder", viewBox: viewBox },
            React.createElement("polygon", { fill: "hsl(210,90%,80%)", points: "0 2,6 2,8 4,16 4,16 6,0 6" }),
            React.createElement("polygon", { fill: "hsl(210,90%,40%)", points: "0 6,16 6,16 14,0 14" })),
        React.createElement("symbol", { id: "more", viewBox: viewBox },
            React.createElement("g", { fill: "currentcolor" },
                React.createElement("circle", { r: "1.5", cx: "3", cy: "8" }),
                React.createElement("circle", { r: "1.5", cx: "8", cy: "8" }),
                React.createElement("circle", { r: "1.5", cx: "13", cy: "8" })))));
}
class Utils {
    /**
     * Format a value as bytes.
     * @param bytes Number of bytes
     * @param decimals Number of decimal places
     */
    static formatBytes(bytes, decimals = 2) {
        const sizes = ["bytes", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB", "RB", "QB"];
        if (bytes === 0)
            return `${bytes} ${sizes[0]}`;
        const KB = 1024;
        const sizeIndex = Math.floor(Math.log(Math.abs(bytes)) / Math.log(KB));
        const value = parseFloat((bytes / (KB ** sizeIndex)).toFixed(decimals));
        const label = sizes[sizeIndex];
        return `${value} ${label}`;
    }
    /**
     * Display a date in a friendly format.
     * @param value Raw date value
     */
    static formatDate(date) {
        return new Intl.DateTimeFormat(this.LOCALE, {
            dateStyle: "medium"
        }).format(date);
    }
    /**
     * Sort two given dates.
     * @param a Date A
     * @param b Date B
     * @param dir Ascending or descending order
     */
    static sortDates(a, b, dir) {
        if (dir === "descending") {
            return b.getTime() - a.getTime();
        }
        return a.getTime() - b.getTime();
    }
    /**
     * Sort two given numbers.
     * @param a Number A
     * @param b Number B
     * @param dir Ascending or descending order
     */
    static sortNumbers(a, b, dir) {
        if (dir === "descending") {
            return b - a;
        }
        return a - b;
    }
    /**
     * Sort two given strings.
     * @param a String A
     * @param b String B
     * @param dir Ascending or descending order
     */
    static sortStrings(a, b, dir) {
        if (dir === "descending") {
            return b.toLowerCase().localeCompare(a.toLowerCase());
        }
        return a.toLowerCase().localeCompare(b.toLowerCase());
    }
}
Utils.LOCALE = "en-US";
;
;