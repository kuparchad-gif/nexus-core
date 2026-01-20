import React, { StrictMode, useEffect, useState } from "https://esm.sh/react";
import { createRoot } from "https://esm.sh/react-dom/client";
import { Flipper, Flipped } from "https://esm.sh/react-flip-toolkit";
import { faker } from "https://esm.sh/@faker-js/faker";

createRoot(document.getElementById("root")!).render(
	<StrictMode>
		<IconSprites />
		<main>
			<FileBrowser />
		</main>
	</StrictMode>
);

/**
 * Generate fake files and folders.
 * @param count Number of items to generate
 */
function generateFakeItems(count: number): FileItem[] {
	const items: FileItem[] = [];

	for (let c = 0; c < count; ++c) {
		const shouldBeDirectory = faker.number.float() < 0.5;
		const users = faker.number.int({ max: 5 });
		const id = faker.string.uuid();
		const sharing: User[] = [];
		const modified = faker.date.past({ years: 2 });

		for (let u = 0; u < users; ++u) {
			// users with whom items will be shared
			const sex = faker.person.sex() as UserSex;
			sharing.push({
				name: faker.person.fullName({ sex }),
				avatar: faker.image.personPortrait({ sex, size: 64 })
			})
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
		} else {
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
	const [files, setFiles] = useState<FileItem[]>(generateFakeItems(5));
	const [sortBy, setSortBy] = useState<SortKey>("name");
	const [sortDir, setSortDir] = useState<SortDir>("ascending");
	const sortOptions: SortOption[] = [
		{ label: "Name", property: "name" },
		{ label: "Sharing", property: "sharing" },
		{ label: "Size", property: "size" },
		{ label: "Modified", property: "modified" },
	];
	const sortFunctions: SortFunctions = {
		name: (a, b, dir) => Utils.sortStrings(a.name, b.name, dir),
		sharing: (a, b, dir) => Utils.sortNumbers(a.sharing.length, b.sharing.length, dir),
		size: (a, b, dir) => Utils.sortNumbers(a.size, b.size, dir),
		modified: (a, b, dir) => Utils.sortDates(a.modified, b.modified, dir)
	};
	const sortSetter = (property: SortKey) => {
		// allow FLIP transitions to run
		setPristine(false);

		if (sortBy === property) {
			setSortDir(sortDir === "ascending" ? "descending" : "ascending");
			return;
		}

		setSortBy(property);
	};

	useEffect(() => {
		const sorted = [...files].sort((a, b) =>
			sortFunctions[sortBy](a as FileItem, b as FileItem, sortDir)
		);

		setFiles(sorted);
	}, [sortDir, sortBy]);

	return (
		<div className="browser" role="grid" aria-rowcount={files.length + 1}>
			<div role="rowgroup">
				<FileBrowserHeader>
					{sortOptions.map((option, i) => (
						<FileBrowserHeaderSort
							key={i}
							{...option}
							sortDir={sortBy === option.property ? sortDir : "none"}
							sortSetter={() => sortSetter(option.property)}
						/>
					))}
					<div className="browser__header-cell" role="columnheader" aria-sort="none">
						<span className="sr-only">More actions</span>
					</div>
				</FileBrowserHeader>
			</div>
			<div className="browser__line"></div>
			<div role="rowgroup">
				<Flipper flipKey={pristine ? "initial" : files.map(file => file.id).join("")}>
					{files.map((file, i) => (
						<Flipped key={file.id} flipId={file.id}>
							<div className="browser__row" role="row" aria-rowindex={i + 2}>
								<FileBrowserRow key={i} {...file} />
							</div>
						</Flipped>
					))}
				</Flipper>
			</div>
		</div>
	);
}
function FileBrowserAvatar({ name, avatar }: FileBrowserAvatarProps) {
	return (
		<div className="avatar">
			<img className="avatar__img" src={avatar} alt={name} title={name} width="24" height="24" />
		</div>
	);
}
function FileBrowserHeader({ children }: FileBrowserHeaderProps) {
	return (
		<div className="browser__header" role="row" aria-rowindex={1}>{children}</div>
	);
}
function FileBrowserHeaderSort({ label, sortDir, sortSetter }: FileBrowserHeaderSortProps) {
	const icons: Record<SortDir, string> = {
		ascending: "caret-up",
		descending: "caret-down",
		none: ""
	};
	const ariaLabel: Record<SortDir, string> = {
		ascending: "(Ascending)",
		descending: "(Descending)",
		none: ""
	};
	const current = sortDir !== "none";
	const buttonClass = "browser__col-button";
	const buttonClasses = `${buttonClass}${current ? ` ${buttonClass}--active` : ""}`;

	return (
		<div className="browser__header-cell" role="columnheader" aria-sort={sortDir}>
			<button className={buttonClasses} type="button" onClick={sortSetter}>
				{label}
				{current && <span aria-label={ariaLabel[sortDir]}><Icon icon={icons[sortDir]} /></span>}
			</button>
		</div>
	);
}
function FileBrowserRow({ name, sharing, size, modified, isDirectory }: FileBrowserRowProps) {
	const icon = isDirectory === true ? "folder" : `file-${name.split(".").pop()}`;
	const isPrivate = sharing.length > 0;
	const avatarLimit = 3;
	const sharingFirst = sharing.slice(0, avatarLimit);
	const sharingRest = Math.max(0, sharing.length - avatarLimit);
	const sharingStatus = isPrivate ? `${sharing.length} user(s)` : "Public";
	const sizeString = Utils.formatBytes(size, 0);
	const modifiedString = Utils.formatDate(modified);

	return (
		<>
			<div className="browser__row-cell" role="gridcell">
				<Icon icon={icon} />
				<div className="browser__row-item">
					<div className="browser__row-name" title={name}>{name}</div>
					<div className="browser__row-details">
						{sharingStatus} &bull; {sizeString} &bull; {modifiedString}
					</div>
				</div>
			</div>
			<div className="browser__row-cell" role="gridcell">
				{isPrivate ? <>
					{sharingFirst.map((user, i) => <FileBrowserAvatar key={i} {...user} />)}
					{sharingRest > 0 && <span className="avatar__count">+{sharingRest}</span>}
				</> : sharingStatus}
			</div>
			<div className="browser__row-cell" role="gridcell">{sizeString}</div>
			<div className="browser__row-cell" role="gridcell">{modifiedString}</div>
			<div className="browser__row-cell" role="gridcell">
				<button className="browser__row-button" type="button" title="More actions">
					<Icon icon="more" />
				</button>
			</div>
		</>
	);
}
function Icon({ icon }: IconProps) {
	const href = `#${icon}`;

	return (
		<svg className="icon" width="16px" height="16px" aria-hidden="true">
			<use href={href} />
		</svg>
	);
}
function IconSprites() {
	const viewBox = "0 0 16 16";

	return (
		<svg width="0" height="0" display="none">
			<symbol id="caret-down" viewBox={viewBox}>
				<polyline fill="none" stroke="currentcolor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" points="4 6,8 10,12 6" />
            </symbol>
			<symbol id="caret-up" viewBox={viewBox}>
				<polyline fill="none" stroke="currentcolor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" points="4 10,8 6,12 10" />
            </symbol>
			<symbol id="file-html" viewBox={viewBox}>
				<polygon fill="hsl(270,90%,90%)" points="1.5 0,9.5 0,14.5 5,14.5 16,1.5 16" />
				<polygon fill="hsl(270,90%,50%)" points="9.5 0,14.5 5,9.5 5" />
				<g fill="none" stroke="hsl(270,90%,50%)" strokeWidth="1">
					<polyline points="7 13,9 8" />
					<polyline points="5.5 9,4 10.5,5.5 12" />
					<polyline points="10.5 9,12 10.5,10.5 12" />
				</g>
            </symbol>
			<symbol id="file-md" viewBox={viewBox}>
				<polygon fill="hsl(0,0%,80%)" points="1.5 0,9.5 0,14.5 5,14.5 16,1.5 16" />
				<polygon fill="hsl(0,0%,40%)" points="9.5 0,14.5 5,9.5 5" />
            </symbol>
			<symbol id="file-txt" viewBox={viewBox}>
				<polygon fill="hsl(210,90%,80%)" points="1.5 0,9.5 0,14.5 5,14.5 16,1.5 16" />
				<polygon fill="hsl(210,90%,40%)" points="9.5 0,14.5 5,9.5 5" />
				<g stroke="hsl(210,90%,40%)" strokeWidth="2">
					<polyline points="4 7,10 7" />
					<polyline points="4 10,8 10" />
					<polyline points="4 13,12 13" />
				</g>
            </symbol>
			<symbol id="folder" viewBox={viewBox}>
				<polygon fill="hsl(210,90%,80%)" points="0 2,6 2,8 4,16 4,16 6,0 6" />
				<polygon fill="hsl(210,90%,40%)" points="0 6,16 6,16 14,0 14" />
            </symbol>
			<symbol id="more" viewBox={viewBox}>
				<g fill="currentcolor">
					<circle r="1.5" cx="3" cy="8" />
					<circle r="1.5" cx="8" cy="8" />
					<circle r="1.5" cx="13" cy="8" />
				</g>
            </symbol>
		</svg>
	);
}

class Utils {
	static LOCALE = "en-US";

	/**
	 * Format a value as bytes.
	 * @param bytes Number of bytes
	 * @param decimals Number of decimal places
	 */
	static formatBytes(bytes: number, decimals: number = 2): string {
		const sizes = ["bytes", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB", "RB", "QB"];

		if (bytes === 0) return `${bytes} ${sizes[0]}`;
		
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
	static formatDate(date: Date) {
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
	static sortDates(a: Date, b: Date, dir?: SortDir) {
		if (dir === "descending") {
			return b.getTime() -  a.getTime();
		}
		return a.getTime() -  b.getTime();
	}
	/**
	 * Sort two given numbers.
	 * @param a Number A
	 * @param b Number B
	 * @param dir Ascending or descending order
	 */
	static sortNumbers(a: number, b: number, dir?: SortDir) {
		if (dir === "descending") {
			return b -  a;
		}
		return a -  b;
	}
	/**
	 * Sort two given strings.
	 * @param a String A
	 * @param b String B
	 * @param dir Ascending or descending order
	 */
	static sortStrings(a: string, b: string, dir?: SortDir) {
		if (dir === "descending") {
			return b.toLowerCase().localeCompare(a.toLowerCase());
		}
		return a.toLowerCase().localeCompare(b.toLowerCase());
	}
}

interface FileItem {
    id: string;
    name: string;
    sharing: User[];
    size: number;
    modified: Date;
    isDirectory?: boolean;
};
interface User {
    name: string;
    avatar: string;
};
type IconProps = {
    icon: string;
};
type FileBrowserAvatarProps = User;
type FileBrowserRowProps = FileItem;
type FileBrowserHeaderProps = {
    children?: React.ReactNode;
};
type FileBrowserHeaderSortProps = {
    label: string;
    sortDir: SortDir;
    sortSetter: () => void;
};
type SortDir = "ascending" | "descending" | "none";
type SortFunctions = Record<SortKey, (a: FileItem, b: FileItem, dir: SortDir) => number>;
type SortKey = "name" | "sharing" | "size" | "modified";
type SortOption = {
    label: string;
    property: SortKey;
};
type UserSex = "female" | "male";